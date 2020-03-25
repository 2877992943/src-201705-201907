# coding:utf-8

import platform
version_python=platform.python_version()
import sys
if version_python.startswith('2'):
  reload(sys)
  sys.setdefaultencoding("utf8")
import os
import pandas as pdd
import json


def writeXLS_fromCSV_toLocal(csv_path,xls_path,fname):
  """
  xxx/xx/xx/xxx.csv
  tmp/
  xxxx
  """
  df = pdd.read_csv(csv_path, encoding='utf-8');#print df.shape
  writer = pdd.ExcelWriter(xls_path+'%s.xlsx'%fname, engine='xlsxwriter')
  df.to_excel(writer, sheet_name='Sheet1', index_label=None, index=None)
  writer.save()


def writeXLS_fromDF_toLocal(df,xls_name,columns=None):
  """
  df
  xxx
  """

  writer = pdd.ExcelWriter('%s.xlsx' % xls_name, engine='xlsxwriter')
  col= df.columns if columns==None else columns
  df.to_excel(writer, sheet_name='Sheet1', index_label=None, index=None,columns=col)
  writer.save()







if __name__=='__main__':
  """  
  #批量 文件夹中的 csv转成 xls 
  ###### transform csv -> xls
  import os
  import pandas as pdd
  filterDict_path='./tmp/'
  xls_path='./xls/'
  for fname in os.listdir(filterDict_path):
    writeXLS_fromCSV_toLocal(filterDict_path+fname,xls_path,fname.strip('csv'))
  """

















