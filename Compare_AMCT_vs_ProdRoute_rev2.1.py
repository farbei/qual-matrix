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
import time
t = time.time()

workdir, outputdir = dirs_address()


def wildCards(df,columns=['OPERATION','ENTITY','PRODUCT','ROUTE']):
    cols = [c for c in columns if c in df.columns]
    wc = lambda c: c.str.replace('*','.*').str.replace('?','.')
    df[cols] = df[cols].astype('str').apply(wc)
    return df


def wild_cards(col):
    return col.str.replace('*','.*').str.replace('?','.')


def restrictMoq(mes):
    if mes['main_moqr'] in ['nan','EMPTY']:
        return ''
    
    comment = []
    uncomment_string = re.sub(r'%[^%]*%','',mes['main_moqr'])
    for field in uncomment_string.strip('/').split('/'):
        if mes['operation'] == field:
            comment.append('oper_rmoq')
        if field.endswith('*'):
            field = field.replace('*','.*').replace('?','.')
            if re.match(field,mes['route']):
                comment.append('route_rmoq')
    
    return ';'.join(comment)


# Restrict L8 to run after limit operations
def cannotFollow(param):
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
def maxCascade(param):
    if 'UDA' in param.keys() and 'MAX_WAFER_COUNT' in param.keys():
        uda_value = mesUDA(param['UDA'])
        if uda_value != 'nan':
            return float(uda_value) > float(param['MAX_WAFER_COUNT'])
    return False


# Check if needed condition ran before operation
def minCondition(param):
    if 'UDA' in param.keys() and 'MIN_WAFER_COUNT' in param.keys():
        uda_value = mesUDA(param['UDA'])
        if uda_value != 'nan':
            return float(uda_value) < float(param['MIN_WAFER_COUNT'])
    return False


def restrictCounter(param):
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


def restrictCounter2(param):

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
    

def closeRow(i,comment,state='Close',dic={'Close':'Open','Down':'Up'}):
    if state in dic.keys():
        mes_table.at[i,'open'] = mes_table['open'][i].replace(dic[state],state)
    else:
        mes_table.at[i,'open'] += '&'+state    
    mes_table.at[i,'close_comment'] += comment+';'

  
def findAmctRow(amct,ref=['OPERATION','PRODUCT','ROUTE','ENTITY']):
    if amct in ref_tables.keys():
        cols = [c for c in ref if c in ref_tables[amct].columns]
        for _, row in ref_tables[amct].iterrows():
            for col in cols:
                if not re.match(row[col],mes_row[col.lower()]): 
                    break
                if col == cols[-1]:
                    return row
    return None
                          

def layerClosed(param, layer='LAYERGROUP'):
#    layer = re.search('LAYERGROUP[^;]*|$',param).group().replace('=','') 
    layer += param[layer] if layer in param.keys() else ''
    return (layer in mes_row.index and mes_row[layer] == 'DOWN')

        
def amctState(entity,param):
    states = ['CH_POR','CH_EX','CH_ASH','CH_SIF',
                  'RECIPE_NAME','RECIPE_CHAMBER'+entity[-1]]
    for par in filter(lambda x: x in param.keys(), states):
        if 'RECIPE' in par:
            return 'CH_SIF' if 'SIF' in param[par] else 'CH_POR', None
        if entity[-1] in param[par]:
            return par, param['CH_ASH'] if 'CH_ASH' in param.keys() else None
    return 'noAmctChamberRef', None


def amct2moduleDic():
    data = pd.read_csv(workdir+'config_file_DB_rev2.csv') 
    values = data[['AMCT_Table','AMCT_Table_72']].values.tolist()
    return dict(zip(data['Module'], values))


def loadMEStable(ceid):
    try:
        df = pd.read_csv(sub_ceid+'_MESTABLE.csv')
        df['open'], df['close_comment'] = 'Up&Open', ''
        col2str = ['operation','oper_process','product','route','main_moqr']
        df[col2str] = df[col2str].astype('str')
        df.update(df.select_dtypes(include=[np.number]).fillna(0))
        return df.fillna('.'), len(df)   
    except OSError as e:
        print(e)
        return [], 0         


def loadAMCTtables(models_list):
    tables_name, process = amct_classes()
    tables, tables_size = dict(zip(process, [{},{}])), []
    
    for val in filter(lambda x: pd.notna(x),models_list):
        for fname in glob.glob('*'+val+'*.csv'):
            process = re.match('(\d{4}|$)', fname).group()
            table = re.search('('+tables_name+')[^\.]*|$', fname).group()
            if table in tables_name:
                data = pd.read_csv(fname) 
                data = data.drop(columns=data.columns[data.isnull().all()])
                data = data.astype('str').apply(wild_cards)
                data = dict_param(data)
                tables_size.append(len(data))
                tables[process][table] = data
    return tables, tables_size
   

def dict_param(df):
    if 'PARAMETER_LIST' in df.columns:
        df['PARAMS'] = df['PARAMETER_LIST'].apply(lambda c: parameterList(c))
    return df


def parameterList(param):
    return dict(tuple(p.split('=',1)) for p in param.split(';') if '=' in p)


def loadSubCeidLegend():
    data = pd.read_csv(workdir+'sub_ceid_legend.csv') 
    data = data.sort_values(by=['module','order'], ascending=True)
    return data['module'].unique(), data


# In Case the 'Layer Allowed' attribute reflect to sub CEID
def fixSubCeid(data): 
    df = data.loc[data['module']==sub_ceid]
    new_ceid = df.loc[mes_row[df['by']].values==df['value'],'new_ceid']
    return new_ceid.values[0] if not new_ceid.empty else mes_row['ceid']   
        

def isAshersDTP(df,ash):
    if ash == None or mes_row['entity'].endswith(('7','8')):
        return False
    
    ashers = [mes_row['entity'][:-1]+x for x in ash.split(',')]
    mask = ( (df['operation'] == mes_row['operation'])
           & (df['product'] == mes_row['product']) 
           & (df['route'] == mes_row['route']) 
           & (df['entity'].isin(ashers)) )
    return mes_table.loc[mask & (df['open'] == 'Up&Open') ].empty


def tool_allowed(tf_row):
    if tf_row is None:
        return False, 'TF=noRef'
    comment = tf_row['COMMENTS'] if 'COMMENTS' in tf_row.index else 'TF'
    return str(tf_row['TOOL_ALLOWED']).upper() == 'TRUE', comment


def group_chambers(entity):
    for _, row in ref_tables['CHAMBER_GROUPS'].iterrows():
        param = row['PARAMS'] #parameterList(row['PARAMETER_LIST'])
        if re.match(row['ENTITY'], entity):
            for pair in param['GROUPS'].split(','):
                if entity[-1] in pair:                    
                   paired = entity[:-1]+pair.strip(entity[-1])
                   pair_rows = mes_table.loc[mes_table['entity']==paired]
                   return pair_rows['sub_availability'].tolist()[0]
    return 'NoPair'           
        
  
def summarizeOperState(df,df_summ):
    if not all(df['entity'].str.endswith(('7','8'))):
        df = df[df['entity'].str.endswith(('7','8'))==False]
    idx = set(df.columns) - set(['entity','close_comment','open'])
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
    for row_idx, mes_row in mes_table.iterrows():
        if not mes_row['processed'] and mes_row['product'] == 'nan':
            drop_rows.append(row_idx)
            continue
        
        if sub_ceid in ceid_needed_fix:
            mes_table.at[row_idx,'ceid'] = fixSubCeid(ceid_legend)  
        elif mes_row['ceid'] != mes_row['f28_ceid']:
            mes_table.at[row_idx,'ceid'] = mes_row['f28_ceid']

        restricted = restrictMoq(mes_row) 
        if restricted:
            closeRow(row_idx,comment=restricted)
                    
        if mes_row['main_availability'] == 'Down':
            closeRow(row_idx,comment='MainDTP',state='Down')
        if mes_row['sub_availability'] == 'Down':
            closeRow(row_idx,comment='ChamberDTP',state='Down')
                        
        ref_tables = tables[mes_row['oper_process'][:4]]
        # TOOL FILTER Table in AMCT
        tf_row = findAmctRow('TOOL_FILTER')
        tf_allowed, tf_comment = tool_allowed(tf_row)
        if not tf_allowed:
            closeRow(row_idx,comment=tf_comment)
                
        f3_row = findAmctRow('F3_SETUP')
        if f3_row is None:
            drop_rows.append(row_idx)
            continue

        #f3_param = f3_row['PARAMS'] #parameterList(f3_row['PARAMETER_LIST'])                
        chamber_state, ashers = amctState(mes_row.entity,f3_row['PARAMS'])
        if chamber_state == 'CH_SIF':
            closeRow(row_idx,comment=chamber_state)  
        if isAshersDTP(mes_table,ashers):
            closeRow(row_idx,comment='NoAshers',state='NoAsh')  
        if restrictCounter(f3_row['PARAMS']):
            closeRow(row_idx,comment='PmCounter')

        lg_row = findAmctRow('LAYERGROUP')
        if lg_row is not None and layerClosed(lg_row['PARAMS']): #'PARAMETER_LIST' 
            closeRow(row_idx,comment='LayerGroup')
               
        ou_row = findAmctRow('OPER_USAGE')
        if ou_row is not None:
            #operusage_param = ou_row['PARAMS'] #parameterList(ou_row['PARAMETER_LIST'])
            if restrictCounter(ou_row['PARAMS']):
                closeRow(row_idx,comment='PmCounter')                        
            if cannotFollow(ou_row['PARAMS']):
                closeRow(row_idx,comment='CannotFollowOper')
                        
        co_row = findAmctRow('CASCADE_OPER')
        if co_row is not None:
            #cascade_param = co_row['PARAMS'] #parameterList(co_row['PARAMETER_LIST'])
            if cannotFollow(co_row['PARAMS']):
                closeRow(row_idx,comment='CannotFollowOper')                        
            if minCondition(co_row['PARAMS']):
                closeRow(row_idx,comment='NeedCond')   
            if maxCascade(co_row['PARAMS']):
                closeRow(row_idx,comment='MaxCascade')
        
        if 'CHAMBER_GROUPS' in ref_tables.keys():
            if group_chambers(mes_row.entity) == 'Down':
                closeRow(row_idx,comment='PairIsDown')
                
        if 'fsui_rules' in ref_tables.keys():
            pass # Need to do something!!!
                            
        if not mes_row['processed'] and mes_table['open'][row_idx] != 'Up&Open':
            drop_rows.append(row_idx)
        
        
    need_columns = ['ceid','operation','oper_short_desc','product','route',
                    'LA24','entity','open','close_comment','Inv','LA6','LA12']   
    df_post = mes_table[need_columns].drop(index=drop_rows)

    sub_ceid_list = list(df_post['ceid'].unique())
    print(sub_ceid,'  ',sub_ceid_list,'  ', len(df_post))
    for sc in sub_ceid_list:
        df_sc = df_post[df_post['ceid']==sc]
        df_sc = df_sc.sort_values(by=['operation','product','route','entity'])
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

    print(time.time() - t)
    
df_summ.to_csv(outputdir+'final_table.csv', index=False)


              
