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
from script_setting import ceid2exclude, dirs_address, amct_classes


workdir, outputdir = dirs_address()

def wildCards(df,columns):
    for col in columns:
        if col in df.columns:
            if str(df[col].unique()) == '[nan]':
                df = df.drop(columns=col)
            else:
                df[col] = df[col].astype('str').str.replace('*','.*').str.replace('?','.')
    return df    


def restrictMoq(uda,param):
    restrict = '('+re.sub(r'\|+','|',re.sub(r'%[^%]*%','',uda).strip('/').replace('/','|'))+')$'
    if not re.match('\d+',str(param)):
        restrict = restrict.replace('*','.*').replace('?','.')
    return re.match(restrict,param)


def restrictMoq2(mes):
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
        uda_value = mesUDA(mes_row,uda='L78GENERICUDA2')
        if uda_value != 'nan':
            return uda_value in param['CANNOT_FOLLOW_OPER']    
    return False     


# Automation L8 UDA 
def mesUDA(mes_row,uda):
    if isinstance(uda,list):
        attr_dic = dict.fromkeys(uda)
        for key in attr_dic:
            attr_name = re.search('([^\']*'+key+'[^\']*|$)',str(mes_row.index)).group()
            if attr_name != '':
                attr_dic[key] = str(int(mes_row[attr_name]))
        return attr_dic
    else:     
        attr_name = re.search('([^\']*'+uda+'[^\']*|$)',str(mes_row.index)).group()
        if attr_name != '':
            return str(int(mes_row[attr_name]))
        else:
            return 'nan'

       
# Check for max cascading wafers allowed to run from operation     
def maxCascade(mes_row,param):
    if 'UDA' in param.keys() and 'MAX_WAFER_COUNT' in param.keys():
        uda_value = mesUDA(mes_row,param['UDA'])
        if uda_value != 'nan':
            return int(uda_value) > int(param['MAX_WAFER_COUNT'])
    
    return False

# Check if needed condition ran before operation
def minCondition(mes_row,param):
    if 'UDA' in param.keys() and 'MIN_WAFER_COUNT' in param.keys():
        uda_value = mesUDA(mes_row,param['UDA'])
        if uda_value != 'nan':
            return int(uda_value) < int(param['MIN_WAFER_COUNT'])
    
    return False


def restrictCounter(mes_row,param):
    if not re.search('UDA|RANGE',str(param.keys())):
        return False
    uda = re.search('[^\']*UDA[^\']*|$',str(param.keys())).group()
    if uda != '':
        uda_value = mesUDA(mes_row,param[uda])
    elif re.search('INJECTION|UPPER_WALL',str(param.keys())):
            uda_value = mesUDA(mes_row,['INJECTOR','UPPERWALL'])
    else:
        return False
    
    if uda_value == 'nan':
        return False
    
    if re.search('RANGES',str(param.keys())):
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


def closeRow(df,i,comment,state='Close'):
    if state == 'Close':
        df.at[i,'open'] = df.at[i,'open'].replace('Open','Close')
    elif state == 'Down':
        df.at[i,'open'] = df.at[i,'open'].replace('Up','Down')
    else:
        df.at[i,'open'] = df.at[i,'open'] + ' & ' + state
        
    if df.at[i,'close_comment'] == '.':
        df.at[i,'close_comment'] = comment   
    else:
        df.at[i,'close_comment'] = df.at[i,'close_comment'] + ';' + comment   


def findAmctRow(mes,amct,join_by=['operation','product','route','entity']):
    cols_match = [x for x in join_by if x.upper() in amct.columns]
    for idx in amct.index:
        for col in cols_match:

            if not re.match(amct[col.upper()][idx],mes[col]):
                break
            if col == cols_match[-1]:
                return amct.iloc[idx], True
    return [], False
                              

def layerClosed(mes_row,param):
    layer_col = re.search('LAYERGROUP[^;]*|$',param).group().replace('=','') 
    if layer_col in mes_table.columns and mes_row[layer_col] == 'DOWN':
        return True
    else:
        False
        
def amctChamberState(entity,f3_param):
    ch = str(entity[-1])
    if 'CH_POR' in f3_param.keys() and re.search(ch,f3_param['CH_POR']):
        etcher = 'por' 
    elif 'CH_ASH' in f3_param.keys() and re.search(ch,f3_param['CH_ASH']):
        etcher = 'ash' 
    elif 'CH_SIF' in f3_param.keys() and re.search(ch,f3_param['CH_SIF']):
        etcher = 'sif' 
    elif 'RECIPE_CHAMBER'+ch in f3_param.keys():
        etcher = 'sif' if re.search('SIF',f3_param['RECIPE_CHAMBER'+ch]) else 'por'
    else:
        etcher = 'noAmctChamberRef'
    # Indicator for ash operation
    ashers = f3_param['CH_ASH'] if 'CH_ASH' in f3_param.keys() and int(ch) < 7 else 'nan'

    return etcher, ashers


def amct2moduleDic():
    data = pd.read_csv(workdir+'config_file_DB_rev2.csv') 
    keys = data['Module']
    values = data[['AMCT_Table','AMCT_Table_72']].values.tolist()
    return dict(zip(keys, values))


def loadMEStable(ceid):
    data = pd.read_csv(sub_ceid+'_MESTABLE.csv') 
    data['open'] = 'Up & Open'
    data['close_comment'] = '.'
    n_rows = len(data)
    if n_rows > 0:
        col2str = ['operation','oper_process','product','route','main_moqr']
        data[col2str] = data[col2str].astype('str')
        data.update(data.select_dtypes(include=[np.number]).fillna(0))
        data.fillna('.', inplace=True)    
    return data, n_rows         


def loadAMCTtables(models_list):
    tables_names, process = amct_classes()
    tables = dict(zip(process, [{},{}]))
    tables_size = []
    
    for idx, val in enumerate(models_list):
        if val != 'X':
            for fname in glob.glob('*'+val+'*.csv'):
                process = re.match('(\d{4}|$)', fname).group()
                table = re.search('('+'|'.join(tables_names)+')[^\.]*|$', fname).group()
                if table in tables_names:
                    data = pd.read_csv(fname) 
                    tables_size.append(len(data))
                    data = wildCards(data,columns=['OPERATION','ENTITY','PRODUCT','ROUTE'])
                    tables[process][table] = data    
    return tables, tables_size
        

def parameterList(parameter_list):
    param_dic = {}
    for param in parameter_list.split(';'):
        if '=' in param:
            key, _, value  = param.partition('=')
            param_dic[key] = value
            
    return param_dic    


def printAMCT2ceid(amct_dic):
    _, process = amct_classes()
    text = ''
    for sub_ceid, amct in amct_dic.items():
        txt = dict(zip(process, ['','']))
        for model in amct:
            if model != 'X':
                for fname in glob.glob('*'+model+'*.csv'):
                    proc = re.match('(\d{4}|$)', fname).group()
                    table = re.search('[^\.]*', fname).group().replace(model,'').replace(proc,'')
                    txt[proc] = txt[proc] + table.replace('__','') + '; '
        text = text + '{0} {3}:\n{4}: {1}\n{5}: {2}\n'.format(sub_ceid, txt[process[0]], txt[process[1]],str(amct),process[0],process[1])            
        #print(text)
        txt_full = text + '\n\n'
        
    with open('amct2ceidList.txt', 'w') as myfile: 
            myfile.write(txt_full) 
                    

def loadSubCeidLegend():
    data = pd.read_csv(workdir+'sub_ceid_legend.csv') 
    return ceid2exclude(), data['module'].unique(), data


# In Case the 'Layer Allowed' attribute reflect to sub CEID
def fixSubCeid(mes_row,data): 
    legend = data[data['module']==mes_row['ceid']].sort_values(by=['order'])

    for index, row in legend.iterrows():
        if re.match(str(row['value']),str(mes_row[row['by']])):
            return row['new_ceid']
    return mes_row['ceid']
        

def isAshersDTP(mes_table,mes_row,ashers):
    for asher in ashers.split(','):
        asher_row = mes_table.loc[(mes_table['operation'] == mes_row['operation']) &
                                  (mes_table['product'] == mes_row['product']) &
                                  (mes_table['route'] == mes_row['route']) &
                                  (mes_table['entity'] == mes_row['entity'][:-1]+asher) ]
        
        if not asher_row.empty and asher_row['open'].item() == 'Up & Open':
            return False
    return True


def summarizeOperState(df_post,df_summ):
    df = df_post.drop(['close_comment'], axis=1)
    df['ceid'] =sub_ceid
    col_list = df.columns.tolist()
    col_list.remove('entity')
    
    pd_count = df.groupby(col_list).agg({'entity':'count'}).reset_index() 
    grouped = pd_count[pd_count['open']=='Up & Open'].groupby(['ceid','operation','oper_short_desc'])   
    df_out = grouped.agg({'entity':['min','max'],'Inv':'sum','LA6':'sum','LA12':'sum','LA24':'sum'}).reset_index() 
    df_out.columns = ['ceid','operation','oper_short_desc','entity_min','entity_max','Inv','LA6','LA12','LA24']
    return df_out if df_summ.empty else pd.concat([df_summ,df_out])

################################################################
os.chdir(workdir)   
amct_dic = amct2moduleDic()
exclude_ceid, ceid_needed_fix, ceid_legend = loadSubCeidLegend()
df_summ = pd.DataFrame({'new' : []})


for sub_ceid, amct in amct_dic.items():
    tables, tables_size = loadAMCTtables(amct)
    mes_table, mes_size = loadMEStable(sub_ceid)
    
    if mes_size == 0 or sub_ceid in exclude_ceid:
        continue
    
    drop_rows = []
    for row_index, mes_row in mes_table.iterrows():
        if mes_row['processed'] == 0 and mes_row['product'] == 'nan':
            drop_rows.append(row_index)
        else:          
            if sub_ceid in ceid_needed_fix:
                mes_table.at[row_index,'ceid'] = fixSubCeid(mes_row,ceid_legend)  
            elif mes_row['ceid'] != mes_row['f28_ceid']:
                mes_table.at[row_index,'ceid'] = mes_row['f28_ceid']
#            elif mes_row['ceid'] != sub_ceid:
#                mes_table.at[row_index,'ceid'] = sub_ceid
                
            if restrictMoq(mes_row['main_moqr'],mes_row['operation']):
                closeRow(mes_table,row_index,comment='MoqOper') 
            if restrictMoq(mes_row['main_moqr'],mes_row['route']):
                closeRow(mes_table,row_index,comment='MoqRoute')
                
#            restricted = restrictMoq2(mes_row) 
#            if restricted:
#                closeRow(mes_table,row_index,comment=restricted)
            
            
            if mes_row['main_availability'] == 'Down':
                closeRow(mes_table,row_index,comment='MainDTP',state='Down')
            if mes_row['sub_availability'] == 'Down':
                closeRow(mes_table,row_index,comment='ChamberDTP',state='Down')
                            
            ref_tables = tables[mes_row['oper_process'][:4]]
            f3_row, found_amct_row_flag = findAmctRow(mes_row,ref_tables['F3_SETUP'])
            if found_amct_row_flag:
                f3_param = parameterList(f3_row['PARAMETER_LIST'])                
                chamber_state, ashers = amctChamberState(mes_row['entity'],f3_param)
                if chamber_state not in ['por','ash']:
                    closeRow(mes_table,row_index,comment=chamber_state)  
                if ashers != 'nan' and isAshersDTP(mes_table,mes_row,ashers):
                    closeRow(mes_table,row_index,comment='NoAshers',state='No Ashers')  
                if restrictCounter(mes_row,f3_param):
                    closeRow(mes_table,row_index,comment='PmCounter')
                                
                # TOOL FILTER Table in AMCT
                if 'TOOL_FILTER' in ref_tables.keys():
                    tf_row, tool_filter_flag = findAmctRow(mes_row,ref_tables['TOOL_FILTER'])
                    if tool_filter_flag and str(tf_row['TOOL_ALLOWED']).lower() == 'false':
                        closeRow(mes_table,row_index,comment='ToolFilter:'+tf_row['COMMENTS'])
    
                if 'LAYERGROUP' in ref_tables.keys():
                    lg_row, layergroup_flag = findAmctRow(mes_row,ref_tables['LAYERGROUP'])
                    if layergroup_flag and layerClosed(mes_row,lg_row['PARAMETER_LIST']): 
                        closeRow(mes_table,row_index,comment='LayerGroup')
                       
                # when ceid use OperUsage table to open/close operation per mes counter.  
                if 'OPER_USAGE' in ref_tables.keys():
                    ou_row, operusage_flag = findAmctRow(mes_row,ref_tables['OPER_USAGE'])
                    if operusage_flag:
                        operusage_param = parameterList(ou_row['PARAMETER_LIST'])
                        if restrictCounter(mes_row,param=operusage_param):
                            closeRow(mes_table,row_index,comment='PmCounter')                        
                        if cannotFollow(mes_row,param=operusage_param):
                            closeRow(mes_table,row_index,comment='CannotFollowOper')
                                
                if 'CASCADE_OPER' in ref_tables.keys():
                    co_row, cascade_oper_flag = findAmctRow(mes_row,ref_tables['CASCADE_OPER'])
                    if cascade_oper_flag:
                        cascade_param = parameterList(co_row['PARAMETER_LIST'])
                        if cannotFollow(mes_row,param=cascade_param):
                            closeRow(mes_table,row_index,comment='CannotFollowOper')                        
                        if minCondition(mes_row,param=cascade_param):
                            closeRow(mes_table,row_index,comment='NeedCond')   
                        if maxCascade(mes_row,param=cascade_param):
                            closeRow(mes_table,row_index,comment='MaxCascade')
                                
                if 'fsui_rules' in ref_tables.keys():
                    do_nothing = 0
                    # Need to do something!!!
                                
            if not found_amct_row_flag or (mes_row['processed'] == 0 and mes_table['open'][row_index] != 'Up & Open'):
                drop_rows.append(row_index)
    
    
    mes_table = mes_table.drop(drop_rows)
    need_columns = ['ceid','operation','oper_short_desc','product','route','LA24',
                    'entity','open','close_comment','Inv','LA6','LA12']    
    df_post = mes_table.filter(need_columns, axis=1)
    df_post.to_csv(outputdir+sub_ceid+'_rev4.csv', index=False)

    sub_ceid_list = list(mes_table['ceid'].unique()) 
    print(sub_ceid,'  ',sub_ceid_list)
    if len(sub_ceid_list) > 1: # Split Table
        for sc in sub_ceid_list:
            df_sc = df_post[df_post['ceid']==sc]
            df_summ = summarizeOperState(df_sc,df_summ)
            df_sc.to_csv(outputdir+sc+'_rev4.csv', index=False)
            print(sc)
    else:
        df_summ = summarizeOperState(df_post,df_summ)

    
    if not all(tables_size) or mes_size == 0:
        txt = sub_ceid + (': mes; ' if mes_size == 0 else ': ')
        for proc, nested_dic in tables.items(): 
            for table, data in nested_dic.items(): 
                if len(data) == 0:
                    txt = txt + proc + ' ' + table + '; ' 
        with open('checkSum.txt', 'a') as myfile: 
            myfile.write(txt + str(dt.now()) + '\n') 


df_summ.to_csv(outputdir+'final_table.csv', index=False)

              
