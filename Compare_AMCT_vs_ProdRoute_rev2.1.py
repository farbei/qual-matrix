# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:34:06 2017

@author: ifarber
"""
### import this
import os
import glob
import re
import pandas as pd 
import numpy as np
from datetime import datetime as dt
from script_setting import dirs_address, amct_classes
#import time

workdir, outputdir = dirs_address()


def wildCards(df,columns=['OPERATION','ENTITY','PRODUCT','ROUTE']):
    for c in columns:
        if c in df.columns:
            if all(df[c].unique() == np.nan):
                df = df.drop(columns=c)
            else:
                df[c] = df[c].astype('str')
                df[c] = df[c].str.replace('*','.*').replace('?','.')
    return df    


def restrictMoq(mes):
    if mes['main_moqr'] in ['nan','EMPTY']:
        return ''
    
    comment = []
    uncomment_string = re.sub(r'%[^%]*%','',mes['main_moqr'])
    for field in uncomment_string.strip('/').split('/'):
        if re.match(r'\d+',field) and mes['operation'] == field:
            comment.append('oper_rmoq')
        if field.endswith('*'):
            field = field.replace('*','.*').replace('?','.')
            if re.match(field,mes['route']):
                comment.append('route_rmoq')
    
    return ';'.join(comment)


# Restrict L8 to run after limit operations
def cannotFollow(mes_row,param):
    if 'CANNOT_FOLLOW_OPER' in param.keys():
        return mesUDA('L78GENERICUDA2') in param['CANNOT_FOLLOW_OPER']    
    return False     


# Automation L8 UDA 
def mesUDA(uda):
    def uda_value(val,attrs):
        attr = re.search('([^\']*'+val+'[^\']*|$)',str(attrs)).group()
        return str(mes_row[attr]) if (val and attr) else 'nan'
    
    if isinstance(uda,str):
        return uda_value(uda,mes_row.index)
    else:
        return {key:uda_value(key,mes_row.index) for key in uda}


# Automation L8 UDA 
def mesUDA2(uda):
    attr = re.search('[^\']*'+uda+'[^\']*',str(mes_row.index))
    return (uda,str(mes_row[attr.group()])) if attr else (uda,'nan')

   
# Check for max cascading wafers allowed to run from operation     
def maxCascade(mes_row,param):
    if 'UDA' in param.keys() and 'MAX_WAFER_COUNT' in param.keys():
        uda_value = mesUDA(param['UDA'])
        if uda_value != 'nan':
            return float(uda_value) > float(param['MAX_WAFER_COUNT'])
    return False


# Check if needed condition ran before operation
def minCondition(mes_row,param):
    if 'UDA' in param.keys() and 'MIN_WAFER_COUNT' in param.keys():
        uda_value = mesUDA(param['UDA'])
        if uda_value != 'nan':
            return float(uda_value) < float(param['MIN_WAFER_COUNT'])
    return False


def restrictCounter(mes_row,param):
    if not re.search('UDA|RANGE',str(param.keys())):
        return False
    uda = re.search('[^\']*UDA[^\']*|$',str(param.keys())).group()
    if uda != '':
        uda_value = mesUDA(param[uda])
    elif re.search('INJECTION|UPPER_WALL',str(param.keys())):
            uda_value = mesUDA(['INJECTOR','UPPERWALL'])
    else:
        return False
    
    if uda_value == 'nan':
        return False
    
    if 'RANGES' in param.keys():
        for counter_limits in param['RANGES'].split(','):
            min_range, _, max_range  = counter_limits.partition('-')
            if float(min_range) < float(uda_value) < float(max_range):
                return False 
        return True
    else:
        min_uda = re.findall('MIN_[PM|UDA|RANGE][^\']*',str(param.keys()))
        lower_limit = '0' if min_uda == [] else param[min_uda[0]]
        max_uda = re.findall('MAX_[PM|UDA|RANGE][^\']*',str(param.keys()))
        for x, y in zip(max_uda, uda_value.values()):
            upper_limit = '9999' if param[x] == 'nan' else param[x]
            if not float(lower_limit) < float(y) < float(upper_limit):
                return True
        return False


def restrictCounter2(mes_row,param):

    if 'MAX_RANGE_INJECTION' in param.keys():
        uda = ['INJECTOR','UPPERWALL']
    else:
        uda = re.search('(PM_UDA|RANGE_UDA|$)',str(param.keys())).group()
        uda = param[uda] if uda else 'nan'
    
    for counter, value in map(mesUDA2,uda):    
        if value == 'nan':
            continue
        if 'RANGES' in param.keys():
            for ranges in param['RANGES'].split(','):
                min_range, _, max_range  = ranges.partition('-')
                if float(min_range) < float(value) < float(max_range):
                    return False 
            return True
        else:
            min_uda = re.search('MIN_[PM|UDA|RANGE]',str(param.keys()))
            lower_limit = param[min_uda] if min_uda else '0'
            max_uda = re.findall('MAX_[PM|UDA|RANGE][^\']*',str(param.keys()))
            for x, y in zip(max_uda, value.values()):
                upper_limit = '9999' if param[x] == 'nan' else param[x]
                if not float(lower_limit) < float(y) < float(upper_limit):
                    return True
            return False
    
    
def closeRow(df,i,comment,state='Close',dic={'Close':'Open','Down':'Up'}):
    if state in dic.keys():
        df.at[i,'open'] = df.at[i,'open'].replace(dic[state],state)
    else:
        df.at[i,'open'] += '&'+state    
    df.at[i,'close_comment'] += comment+';'

  
def findAmctRow(amct,join_by=['operation','product','route','entity']):
    cols_match = [x for x in join_by if x.upper() in amct.columns]
    for idx in amct.index:
        for col in cols_match:
            if not re.match(amct[col.upper()][idx],mes_row[col]):
                break
            if col == cols_match[-1]:
                return amct.iloc[idx], True
    return [], False
                              

def layerClosed(mes_row,param):
    layer = re.search('LAYERGROUP[^;]*|$',param).group().replace('=','') 
    return (layer in mes_row.index and mes_row[layer] == 'DOWN')

        
def amctState(entity,f3_param):
    for par in ['CH_POR','CH_EX','CH_ASH','CH_SIF']:
        if par in f3_param.keys() and entity[-1] in f3_param[par]:
            if 'CH_ASH' in f3_param.keys() and entity[-1] < '7':
                return par, f3_param['CH_ASH']
            return par, 'nan' 
     
    for par in ['RECIPE_NAME','RECIPE_CHAMBER'+entity[-1]]:
        if par in f3_param.keys() and f3_param[par]:
            etcher = 'CH_SIF' if 'SIF' in f3_param[par] else 'CH_POR'
            return etcher, 'nan' 

    return 'noAmctChamberRef', 'nan'


def amct2moduleDic():
    data = pd.read_csv(workdir+'config_file_DB_rev2.csv') 
    keys = data['Module']
    values = data[['AMCT_Table','AMCT_Table_72']].values.tolist()
    return dict(zip(keys, values))


def loadMEStable(ceid):
    try:
        data = pd.read_csv(sub_ceid+'_MESTABLE.csv')
    except OSError as e:
        print(e)
        return [], 0
        
    data['open'], data['close_comment'] = 'Up&Open', ''
     
    n_rows = len(data)
    if n_rows > 0:
        col2str = ['operation','oper_process','product','route','main_moqr']
        data[col2str] = data[col2str].astype('str')
        data.update(data.select_dtypes(include=[np.number]).fillna(0))
        data.fillna('.', inplace=True)    
    return data, n_rows         


def loadAMCTtables(models_list):
    tables_name, process = amct_classes()
    tables, tables_size = dict(zip(process, [{},{}])), []
    
    for idx, val in enumerate(models_list):
        if val != 'X':
            for fname in glob.glob('*'+val+'*.csv'):
                process = re.match('(\d{4}|$)', fname).group()
                table = re.search('('+tables_name+')[^\.]*|$', fname).group()
                if table in tables_name:
                    data = pd.read_csv(fname) 
                    tables_size.append(len(data))
                    tables[process][table] = wildCards(data)    
    return tables, tables_size
   

def parameterList(param):
    return dict(tuple(p.split('=',1)) for p in param.split(';') if '=' in p)


def loadSubCeidLegend():
    data = pd.read_csv(workdir+'sub_ceid_legend.csv') 
    data = data.sort_values(by=['module','order'], ascending=True)
    return data['module'].unique(), data


# In Case the 'Layer Allowed' attribute reflect to sub CEID
def fixSubCeid(data): 
    for index, row in data.loc[data['module']==sub_ceid].iterrows():
        if mes_row[row['by']] == row['value']:
            return row['new_ceid']
    return mes_row['ceid']
        

def isAshersDTP(df,ashers):
    oper, prod, rout, ent = mes_row[['operation','product','route','entity']]
    mask = (df.operation == oper) & (df.product == prod) & (df.route == rout)
    for asher in ashers.split(','):
        asher = mes_table.loc[mask & (df.entity == ent[:-1]+asher)]
        if not asher.empty and asher['open'].item() == 'Up&Open':
            return False
    return True 


def summarizeOperState(df,df_summ):
    idx = [c for c in df.columns if c not in ['entity','close_comment','open']]
    table = df.pivot_table(values='entity', index=idx, columns=['open'],
                           aggfunc={'entity': 'count'}).reset_index()
    
    if 'Up&Open' in table.columns:
        table = table.rename(columns={'Up&Open': 'entity_'})
    else:
        table['entity_'] = 0
    grouped = table.groupby(['ceid','operation','oper_short_desc']
                            , as_index=False)
    df_out = grouped.agg({'entity_':['min','max'],
                          'Inv':'sum','LA6':'sum','LA12':'sum','LA24':'sum'})
    df_out.columns = ["".join(x) for x in df_out.columns.ravel()]

    return df_out if df_summ.empty else pd.concat([df_summ,df_out])

################################################################
os.chdir(workdir)   
amct_dic = amct2moduleDic()
ceid_needed_fix, ceid_legend = loadSubCeidLegend()
df_summ = pd.DataFrame({'new' : []})

for sub_ceid, amct in amct_dic.items():

    tables, tables_size = loadAMCTtables(amct)
    mes_table, mes_size = loadMEStable(sub_ceid)
    
    if mes_size == 0:
        continue
    
    drop_rows = []
    for row_index, mes_row in mes_table.iterrows():
        if not mes_row['processed'] and mes_row['product'] == 'nan':
            drop_rows.append(row_index)
        else:          
            if sub_ceid in ceid_needed_fix:
                mes_table.at[row_index,'ceid'] = fixSubCeid(ceid_legend)  
            elif mes_row['ceid'] != mes_row['f28_ceid']:
                mes_table.at[row_index,'ceid'] = mes_row['f28_ceid']
                
            restricted = restrictMoq(mes_row) 
            if restricted:
                closeRow(mes_table,row_index,comment=restricted)
                        
            if mes_row['main_availability'] == 'Down':
                closeRow(mes_table,row_index,comment='MainDTP',state='Down')
            if mes_row['sub_availability'] == 'Down':
                closeRow(mes_table,row_index,comment='ChamberDTP',state='Down')
                            
            ref_tables = tables[mes_row['oper_process'][:4]]
            f3_row, found_amct_row_flag = findAmctRow(ref_tables['F3_SETUP'])
            if found_amct_row_flag:
                f3_param = parameterList(f3_row['PARAMETER_LIST'])                
                chamber_state, ashers = amctState(mes_row['entity'],f3_param)
                if chamber_state not in ['CH_POR','CH_EX','CH_ASH']:
                    closeRow(mes_table,row_index,comment=chamber_state)  
                if ashers != 'nan' and isAshersDTP(mes_table,ashers):
                    closeRow(mes_table,row_index,comment='NoAshers',state='No Ashers')  
                if restrictCounter(mes_row,f3_param):
                    closeRow(mes_table,row_index,comment='PmCounter')
                                
                # TOOL FILTER Table in AMCT
                if 'TOOL_FILTER' in ref_tables.keys():
                    tf_row, tool_filter_flag = findAmctRow(ref_tables['TOOL_FILTER'])
                    if tool_filter_flag and str(tf_row['TOOL_ALLOWED']).lower() == 'false':
                        closeRow(mes_table,row_index,comment='ToolFilter:'+tf_row['COMMENTS'])
    
                if 'LAYERGROUP' in ref_tables.keys():
                    lg_row, layergroup_flag = findAmctRow(ref_tables['LAYERGROUP'])
                    if layergroup_flag and layerClosed(mes_row,lg_row['PARAMETER_LIST']): 
                        closeRow(mes_table,row_index,comment='LayerGroup')
                       
                # when ceid use OperUsage table to open/close operation per mes counter.  
                if 'OPER_USAGE' in ref_tables.keys():
                    ou_row, operusage_flag = findAmctRow(ref_tables['OPER_USAGE'])
                    if operusage_flag:
                        operusage_param = parameterList(ou_row['PARAMETER_LIST'])
                        if restrictCounter(mes_row,param=operusage_param):
                            closeRow(mes_table,row_index,comment='PmCounter')                        
                        if cannotFollow(mes_row,param=operusage_param):
                            closeRow(mes_table,row_index,comment='CannotFollowOper')
                                
                if 'CASCADE_OPER' in ref_tables.keys():
                    co_row, cascade_oper_flag = findAmctRow(ref_tables['CASCADE_OPER'])
                    if cascade_oper_flag:
                        cascade_param = parameterList(co_row['PARAMETER_LIST'])
                        if cannotFollow(mes_row,param=cascade_param):
                            closeRow(mes_table,row_index,comment='CannotFollowOper')                        
                        if minCondition(mes_row,param=cascade_param):
                            closeRow(mes_table,row_index,comment='NeedCond')   
                        if maxCascade(mes_row,param=cascade_param):
                            closeRow(mes_table,row_index,comment='MaxCascade')
                                
                if 'fsui_rules' in ref_tables.keys():
                    pass # Need to do something!!!
                                
            if not found_amct_row_flag or (
                    not mes_row['processed'] 
                    and mes_table['open'][row_index] != 'Up & Open'):
                drop_rows.append(row_index)
    
    
    need_columns = ['ceid','operation','oper_short_desc','product','route','LA24',
                    'entity','open','close_comment','Inv','LA6','LA12']   
    df_post = mes_table[need_columns].drop(index=drop_rows)

    sub_ceid_list = list(df_post['ceid'].unique())
    print(sub_ceid,'  ',sub_ceid_list,'  ', len(df_post))
    for sc in sub_ceid_list:
        df_sc = df_post[df_post['ceid']==sc]
        df_summ = summarizeOperState(df_sc,df_summ)
        df_sc.to_csv(outputdir+sc+'_rev4.csv', index=False)
    
    
    if not all(tables_size) or mes_size == 0:
        txt = sub_ceid + (': mes; ' if mes_size == 0 else ': ')
        for proc, nested_dic in tables.items(): 
            for table, data in nested_dic.items(): 
                if len(data) == 0:
                    txt = txt + proc + ' ' + table + '; ' 
        with open('checkSum.txt', 'a') as myfile: 
            myfile.write(txt + str(dt.now()) + '\n') 


df_summ.to_csv(outputdir+'final_table.csv', index=False)

              
