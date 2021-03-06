
# coding: utf-8

# In[1]:


import csv 
import json 
import pandas as pd
from pandas import DataFrame as df
import sklearn 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score ,confusion_matrix,precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB 
import boto
import boto3
from boto.s3.key import Key
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import urllib
from urllib import request
import argparse
import logging
import sys


# In[2]:


#create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) #setting level to debug

#creates file handler which logs all the level messages
fh = logging.FileHandler('Log_filedelay.log' , mode = 'w')
fh.setLevel(logging.DEBUG) #setting level to handler to debug

formatter = logging.Formatter('[%(asctime)s] %(levelname)8s --- %(message)s' + 
                             '(%(filename)s:%(lineno)s)' ,datefmt = '%Y-%m-%d %H:%M:%S')

if (logger.hasHandlers()):
    logger.handlers.clear()
    
#if handler_console is none :
handler_console = logging.StreamHandler(stream = sys.stdout)
#then add it back 
handler_console.setFormatter(formatter)
handler_console.setLevel(logging.INFO)
logger.addHandler(handler_console)

fh.setFormatter(formatter)
logger.addHandler(fh)


# In[3]:


url ='https://storage.googleapis.com/team3/final_flight_delay.csv'
logging.info("Dataset from the Google cloud into csv format is loaded")
df = urllib.request.urlopen(url)

df  = pd.read_csv(df)


# In[4]:


#def cleaning(data):
data = df.fillna(method='bfill',axis=0).fillna('0')
logging.info("Cleaning of the data is performed")
#    return data
#cleaning(df)


# In[5]:


#def featureeng(data):
    
logging.info("Feature Engineering is performed on dataset, some extra coloumns are added and some are modified")
uc = data.UniqueCarrier
uc = uc.replace('9E','1')
uc = uc.replace('AA','2')
uc = uc.replace('AQ','3')
uc = uc.replace('AS','4')
uc = uc.replace('B6','5')
uc = uc.replace('CO','6')
uc = uc.replace('DL','7')
uc = uc.replace('EV','8')
uc = uc.replace('F9','9')
uc = uc.replace('FE','10')
uc = uc.replace('HA','11')
uc = uc.replace('MQ','12')
uc = uc.replace('NW','13')
uc = uc.replace('OH','14')
uc = uc.replace('OO','15')
uc = uc.replace('UA','16')
uc = uc.replace('US','17')
uc = uc.replace('WN','18')
uc = uc.replace('XE','19')
uc = uc.replace('YV','20')
uc = uc.replace('FL','21')

data['uniquecarrier_int']=uc

origin = data['Origin']
origin = origin.replace('ABE','1')
origin = origin.replace('ABI','2')
origin = origin.replace('ABQ','3')
origin = origin.replace('ABY','4')
origin = origin.replace('ACT','5')
origin = origin.replace('ACV','6')
origin = origin.replace('ACY','7')
origin = origin.replace('ADK','8')
origin = origin.replace('ADQ','9')
origin = origin.replace('AEX','10')
origin = origin.replace('AGS','11')
origin = origin.replace('AKN','12')
origin = origin.replace('ALB','13')
origin = origin.replace('ALO','14')
origin = origin.replace('AMA','15')
origin = origin.replace('ANC','16')
origin = origin.replace('APF','17')
origin = origin.replace('ASE','18')
origin = origin.replace('ATL','19')
origin = origin.replace('ATW','20')
origin = origin.replace('AUS','21')
origin = origin.replace('AVL','22')
origin = origin.replace('AVP','23')
origin = origin.replace('AZO','24')
origin = origin.replace('BDL','25')
origin = origin.replace('BET','26')
origin = origin.replace('BFL','27')
origin = origin.replace('BGM','28')
origin = origin.replace('BGR','29')
origin = origin.replace('BHM','30')
origin = origin.replace('BIL','31')
origin = origin.replace('BIS','32')
origin = origin.replace('BLI','33')
origin = origin.replace('BMI','34')
origin = origin.replace('BNA','35')
origin = origin.replace('BOI','36')
origin = origin.replace('BOS','37')
origin = origin.replace('BPT','38')
origin = origin.replace('BQK','39')
origin = origin.replace('BQN','40')
origin = origin.replace('BRO','41')
origin = origin.replace('BRW','42')
origin = origin.replace('BTM','43')
origin = origin.replace('BTR','44')
origin = origin.replace('BTV','45')
origin = origin.replace('BUF','46')
origin = origin.replace('BUR','47')
origin = origin.replace('BWI','48')
origin = origin.replace('BZN','49')
origin = origin.replace('CAE','50')
origin = origin.replace('CAK','51')
origin = origin.replace('CDV','52')
origin = origin.replace('CEC','53')
origin = origin.replace('CHA','54')
origin = origin.replace('CHO','55')
origin = origin.replace('CHS','56')
origin = origin.replace('CIC','57')
origin = origin.replace('CID','58')
origin = origin.replace('CLD','59')
origin = origin.replace('CLE','60')
origin = origin.replace('CLL','61')
origin = origin.replace('CLT','62')
origin = origin.replace('CMH','63')
origin = origin.replace('CMI','64')
origin = origin.replace('CMX','65')
origin = origin.replace('COD','66')
origin = origin.replace('CPR','67')
origin = origin.replace('CRP','68')
origin = origin.replace('CRW','69')
origin = origin.replace('CSG','70')
origin = origin.replace('CVG','71')
origin = origin.replace('CWA','72')
origin = origin.replace('DAB','73')
origin = origin.replace('DAL','74')
origin = origin.replace('DAY','75')
origin = origin.replace('DBQ','76')
origin = origin.replace('DCA','77')
origin = origin.replace('DEN','78')
origin = origin.replace('DFW','79')
origin = origin.replace('DHN','80')
origin = origin.replace('DLG','81')
origin = origin.replace('DLH','82')
origin = origin.replace('DRO','83')
origin = origin.replace('DSM','84')
origin = origin.replace('DTW','85')
origin = origin.replace('EGE','86')
origin = origin.replace('EKO','87')
origin = origin.replace('ELM','88')
origin = origin.replace('ELP','89')
origin = origin.replace('ERI','90')
origin = origin.replace('EUG','91')
origin = origin.replace('EVV','92')
origin = origin.replace('EWR','93')
origin = origin.replace('EYW','94')
origin = origin.replace('FAI','95')
origin = origin.replace('FAR','96')
origin = origin.replace('FAT','97')
origin = origin.replace('FAY','98')
origin = origin.replace('FCA','99')
origin = origin.replace('FLG','100')
origin = origin.replace('FLL','101')
origin = origin.replace('FLO','102')
origin = origin.replace('FNT','103')
origin = origin.replace('FSD','104')
origin = origin.replace('FSM','105')
origin = origin.replace('FWA','106')
origin = origin.replace('GEG','107')
origin = origin.replace('GFK','108')
origin = origin.replace('GGG','109')
origin = origin.replace('GJT','110')
origin = origin.replace('GNV','111')
origin = origin.replace('GPT','112')
origin = origin.replace('GRB','113')
origin = origin.replace('GRK','114')
origin = origin.replace('GRR','115')
origin = origin.replace('GSO','116')
origin = origin.replace('GSP','117')
origin = origin.replace('GTF','118')
origin = origin.replace('GTR','119')
origin = origin.replace('GUC','120')
origin = origin.replace('HDN','121')
origin = origin.replace('HLN','122')
origin = origin.replace('HNL','123')
origin = origin.replace('HOU','124')
origin = origin.replace('HPN','125')
origin = origin.replace('HRL','126')
origin = origin.replace('HSV','127')
origin = origin.replace('HTS','128')
origin = origin.replace('IAD','129')
origin = origin.replace('IAH','130')
origin = origin.replace('ICT','131')
origin = origin.replace('IDA','132')
origin = origin.replace('ILG','133')
origin = origin.replace('ILM','134')
origin = origin.replace('IND','135')
origin = origin.replace('IPL','136')
origin = origin.replace('ISO','137')
origin = origin.replace('ISP','138')
origin = origin.replace('ITO','139')
origin = origin.replace('IYK','140')
origin = origin.replace('JAC','141')
origin = origin.replace('JAN','142')
origin = origin.replace('JAX','143')
origin = origin.replace('JFK','144')
origin = origin.replace('JNU','145')
origin = origin.replace('KOA','146')
origin = origin.replace('KTN','147')
origin = origin.replace('LAN','148')
origin = origin.replace('LAS','149')
origin = origin.replace('LAN','150')
origin = origin.replace('LAX','151')
origin = origin.replace('LBB','152')
origin = origin.replace('LCH','153')
origin = origin.replace('LEX','154')
origin = origin.replace('LFT','155')
origin = origin.replace('LGA','156')
origin = origin.replace('LGB','157')
origin = origin.replace('LIH','158')
origin = origin.replace('LIT','159')
origin = origin.replace('LNK','160')
origin = origin.replace('LRD','161')
origin = origin.replace('LSE','162')
origin = origin.replace('LWS','163')
origin = origin.replace('LYH','164')
origin = origin.replace('MAF','165')
origin = origin.replace('MCI','166')
origin = origin.replace('MCN','167')
origin = origin.replace('MCO','168')
origin = origin.replace('MDT','169')
origin = origin.replace('MDW','170')
origin = origin.replace('MEI','171')
origin = origin.replace('MEM','172')
origin = origin.replace('MFE','173')
origin = origin.replace('MFR','174')
origin = origin.replace('MGM','175')
origin = origin.replace('MHT','176')
origin = origin.replace('MIA','177')
origin = origin.replace('MKE','178')
origin = origin.replace('MLB','179')
origin = origin.replace('MLI','180')
origin = origin.replace('MLU','181')
origin = origin.replace('MOB','182')
origin = origin.replace('MOD','183')
origin = origin.replace('MOT','184')
origin = origin.replace('MQT','185')
origin = origin.replace('MRY','186')
origin = origin.replace('MSN','187')
origin = origin.replace('MSO','188')
origin = origin.replace('MSP','189')
origin = origin.replace('MSY','190')
origin = origin.replace('MTH','191')
origin = origin.replace('MTJ','192')
origin = origin.replace('MYR','193')
origin = origin.replace('OAJ','194')
origin = origin.replace('OAK','195')
origin = origin.replace('OGG','196')
origin = origin.replace('OKC','197')
origin = origin.replace('OMA','198')
origin = origin.replace('OME','199')
origin = origin.replace('ONT','200')
origin = origin.replace('ORD','201')
origin = origin.replace('ORF','202')
origin = origin.replace('OTZ','203')
origin = origin.replace('OXR','204')
origin = origin.replace('PBI','205')
origin = origin.replace('PDX','206')
origin = origin.replace('PFN','207')
origin = origin.replace('PHF','208')
origin = origin.replace('PHL','209')
origin = origin.replace('PIA','210')
origin = origin.replace('PIH','211')
origin = origin.replace('PLN','212')
origin = origin.replace('PNS','213')
origin = origin.replace('PSC','214')
origin = origin.replace('PSE','215')
origin = origin.replace('PSG','216')
origin = origin.replace('PSP','217')
origin = origin.replace('PVD','218')
origin = origin.replace('PWM','219')
origin = origin.replace('RAP','220')
origin = origin.replace('RDD','221')
origin = origin.replace('RDM','222')
origin = origin.replace('RDU','223')
origin = origin.replace('RFD','224')
origin = origin.replace('RHI','225')
origin = origin.replace('RIC','226')
origin = origin.replace('RNO','227')
origin = origin.replace('ROA','228')
origin = origin.replace('ROC','229')
origin = origin.replace('RST','230')
origin = origin.replace('RSW','231')
origin = origin.replace('SAN','232')
origin = origin.replace('SAT','233')
origin = origin.replace('SAV','234')
origin = origin.replace('SBA','235')
origin = origin.replace('SBN','236')
origin = origin.replace('SBP','237')
origin = origin.replace('SCC','238')
origin = origin.replace('SCE','239')
origin = origin.replace('SDF','240')
origin = origin.replace('SEA','241')
origin = origin.replace('SFO','242')
origin = origin.replace('SGF','243')
origin = origin.replace('SGU','244')
origin = origin.replace('SHV','245')
origin = origin.replace('SIT','246')
origin = origin.replace('SJC','247')
origin = origin.replace('SJT','248')
origin = origin.replace('SJU','249')
origin = origin.replace('SLC','250')
origin = origin.replace('SMF','251')
origin = origin.replace('SMX','252')
origin = origin.replace('SNA','253')
origin = origin.replace('SPI','254')
origin = origin.replace('SPS','255')
origin = origin.replace('SRQ','256')
origin = origin.replace('STL','257')
origin = origin.replace('STT','258')
origin = origin.replace('STX','259')
origin = origin.replace('SUN','260')
origin = origin.replace('SUX','261')
origin = origin.replace('SWF','262')
origin = origin.replace('SYR','263')
origin = origin.replace('TEX','264')
origin = origin.replace('TLH','265')
origin = origin.replace('TOL','266')
origin = origin.replace('TPA','267')
origin = origin.replace('TRI','268')
origin = origin.replace('TTN','269')
origin = origin.replace('TUL','270')
origin = origin.replace('TUP','271')
origin = origin.replace('TUS','272')
origin = origin.replace('TVC','273')
origin = origin.replace('TWF','274')
origin = origin.replace('TXK','275')
origin = origin.replace('TYR','276')
origin = origin.replace('TYS','277')
origin = origin.replace('VLD','278')
origin = origin.replace('VPS','279')
origin = origin.replace('WRG','280')
origin = origin.replace('XNA','281')
origin = origin.replace('YAK','282')
origin = origin.replace('YUM','283')
origin = origin.replace('ACK','284')
origin = origin.replace('APF','285')
origin = origin.replace('BJI','286')
origin = origin.replace('PHX','287')
origin = origin.replace('CDC','288')
origin = origin.replace('EWN','289')
origin = origin.replace('COS','290')
origin = origin.replace('GCC','291')
origin = origin.replace('GST','292')
origin = origin.replace('HHH','293')
origin = origin.replace('INL','294')
origin = origin.replace('ITH','295')
origin = origin.replace('LAW','296')
origin = origin.replace('LMT','297')
origin = origin.replace('OTH','298')
origin = origin.replace('PIR','299')
origin = origin.replace('PIT','300')
origin = origin.replace('PMD','301')
origin = origin.replace('PUB','302')
origin = origin.replace('RKS','303')
origin = origin.replace('ROW','304')
origin = origin.replace('SLE','305')
origin = origin.replace('WYS','306')
origin = origin.replace('YKM','307')
origin = origin.replace('BFF','308')
origin = origin.replace('CDC','309')
origin = origin.replace('OGD','310')
origin = origin.replace('PVU','311')
origin = origin.replace('MBS','312')
origin = origin.replace('CYS','313')
origin = origin.replace('MKG','314')
origin = origin.replace('LWB','315')

data['origin_int'] = origin

dest = data['Dest']
dest = dest.replace('ABE','1')
dest = dest.replace('ABI','2')
dest = dest.replace('ABQ','3')
dest = dest.replace('ABY','4')
dest = dest.replace('ACT','5')
dest = dest.replace('ACV','6')
dest = dest.replace('ACY','7')
dest = dest.replace('ADK','8')
dest = dest.replace('ADQ','9')
dest = dest.replace('AEX','10')
dest = dest.replace('AGS','11')
dest = dest.replace('AKN','12')
dest = dest.replace('ALB','13')
dest = dest.replace('ALO','14')
dest = dest.replace('AMA','15')
dest = dest.replace('ANC','16')
dest = dest.replace('APF','17')
dest = dest.replace('ASE','18')
dest = dest.replace('ATL','19')
dest = dest.replace('ATW','20')
dest = dest.replace('AUS','21')
dest = dest.replace('AVL','22')
dest = dest.replace('AVP','23')
dest = dest.replace('AZO','24')
dest = dest.replace('BDL','25')
dest = dest.replace('BET','26')
dest = dest.replace('BFL','27')
dest = dest.replace('BGM','28')
dest = dest.replace('BGR','29')
dest = dest.replace('BHM','30')
dest = dest.replace('BIL','31')
dest = dest.replace('BIS','32')
dest = dest.replace('BLI','33')
dest = dest.replace('BMI','34')
dest = dest.replace('BNA','35')
dest = dest.replace('BOI','36')
dest = dest.replace('BOS','37')
dest = dest.replace('BPT','38')
dest = dest.replace('BQK','39')
dest = dest.replace('BQN','40')
dest = dest.replace('BRO','41')
dest = dest.replace('BRW','42')
dest = dest.replace('BTM','43')
dest = dest.replace('BTR','44')
dest = dest.replace('BTV','45')
dest = dest.replace('BUF','46')
dest = dest.replace('BUR','47')
dest = dest.replace('BWI','48')
dest = dest.replace('BZN','49')
dest = dest.replace('CAE','50')
dest = dest.replace('CAK','51')
dest = dest.replace('CDV','52')
dest = dest.replace('CEC','53')
dest = dest.replace('CHA','54')
dest = dest.replace('CHO','55')
dest = dest.replace('CHS','56')
dest = dest.replace('CIC','57')
dest = dest.replace('CID','58')
dest = dest.replace('CLD','59')
dest = dest.replace('CLE','60')
dest = dest.replace('CLL','61')
dest = dest.replace('CLT','62')
dest = dest.replace('CMH','63')
dest = dest.replace('CMI','64')
dest = dest.replace('CMX','65')
dest = dest.replace('COD','66')
dest = dest.replace('CPR','67')
dest = dest.replace('CRP','68')
dest = dest.replace('CRW','69')
dest = dest.replace('CSG','70')
dest = dest.replace('CVG','71')
dest = dest.replace('CWA','72')
dest = dest.replace('DAB','73')
dest = dest.replace('DAL','74')
dest = dest.replace('DAY','75')
dest = dest.replace('DBQ','76')
dest = dest.replace('DCA','77')
dest = dest.replace('DEN','78')
dest = dest.replace('DFW','79')
dest = dest.replace('DHN','80')
dest = dest.replace('DLG','81')
dest = dest.replace('DLH','82')
dest = dest.replace('DRO','83')
dest = dest.replace('DSM','84')
dest = dest.replace('DTW','85')
dest = dest.replace('EGE','86')
dest = dest.replace('EKO','87')
dest = dest.replace('ELM','88')
dest = dest.replace('ELP','89')
dest = dest.replace('ERI','90')
dest = dest.replace('EUG','91')
dest = dest.replace('EVV','92')
dest = dest.replace('EWR','93')
dest = dest.replace('EYW','94')
dest = dest.replace('FAI','95')
dest = dest.replace('FAR','96')
dest = dest.replace('FAT','97')
dest = dest.replace('FAY','98')
dest = dest.replace('FCA','99')
dest = dest.replace('FLG','100')
dest = dest.replace('FLL','101')
dest = dest.replace('FLO','102')
dest = dest.replace('FNT','103')
dest = dest.replace('FSD','104')
dest = dest.replace('FSM','105')
dest = dest.replace('FWA','106')
dest = dest.replace('GEG','107')
dest = dest.replace('GFK','108')
dest = dest.replace('GGG','109')
dest = dest.replace('GJT','110')
dest = dest.replace('GNV','111')
dest = dest.replace('GPT','112')
dest = dest.replace('GRB','113')
dest = dest.replace('GRK','114')
dest = dest.replace('GRR','115')
dest = dest.replace('GSO','116')
dest = dest.replace('GSP','117')
dest = dest.replace('GTF','118')
dest = dest.replace('GTR','119')
dest = dest.replace('GUC','120')
dest = dest.replace('HDN','121')
dest = dest.replace('HLN','122')
dest = dest.replace('HNL','123')
dest = dest.replace('HOU','124')
dest = dest.replace('HPN','125')
dest = dest.replace('HRL','126')
dest = dest.replace('HSV','127')
dest = dest.replace('HTS','128')
dest = dest.replace('IAD','129')
dest = dest.replace('IAH','130')
dest = dest.replace('ICT','131')
dest = dest.replace('IDA','132')
dest = dest.replace('ILG','133')
dest = dest.replace('ILM','134')
dest = dest.replace('IND','135')
dest = dest.replace('IPL','136')
dest = dest.replace('ISO','137')
dest = dest.replace('ISP','138')
dest = dest.replace('ITO','139')
dest = dest.replace('IYK','140')
dest = dest.replace('JAC','141')
dest = dest.replace('JAN','142')
dest = dest.replace('JAX','143')
dest = dest.replace('JFK','144')
dest = dest.replace('JNU','145')
dest = dest.replace('KOA','146')
dest = dest.replace('KTN','147')
dest = dest.replace('LAN','148')
dest = dest.replace('LAS','149')
dest = dest.replace('LAN','150')
dest = dest.replace('LAX','151')
dest = dest.replace('LBB','152')
dest = dest.replace('LCH','153')
dest = dest.replace('LEX','154')
dest = dest.replace('LFT','155')
dest = dest.replace('LGA','156')
dest = dest.replace('LGB','157')
dest = dest.replace('LIH','158')
dest = dest.replace('LIT','159')
dest = dest.replace('LNK','160')
dest = dest.replace('LRD','161')
dest = dest.replace('LSE','162')
dest = dest.replace('LWS','163')
dest = dest.replace('LYH','164')
dest = dest.replace('MAF','165')
dest = dest.replace('MCI','166')
dest = dest.replace('MCN','167')
dest = dest.replace('MCO','168')
dest = dest.replace('MDT','169')
dest = dest.replace('MDW','170')
dest = dest.replace('MEI','171')
dest = dest.replace('MEM','172')
dest = dest.replace('MFE','173')
dest = dest.replace('MFR','174')
dest = dest.replace('MGM','175')
dest = dest.replace('MHT','176')
dest = dest.replace('MIA','177')
dest = dest.replace('MKE','178')
dest = dest.replace('MLB','179')
dest = dest.replace('MLI','180')
dest = dest.replace('MLU','181')
dest = dest.replace('MOB','182')
dest = dest.replace('MOD','183')
dest = dest.replace('MOT','184')
dest = dest.replace('MQT','185')
dest = dest.replace('MRY','186')
dest = dest.replace('MSN','187')
dest = dest.replace('MSO','188')
dest = dest.replace('MSP','189')
dest = dest.replace('MSY','190')
dest = dest.replace('MTH','191')
dest = dest.replace('MTJ','192')
dest = dest.replace('MYR','193')
dest = dest.replace('OAJ','194')
dest = dest.replace('OAK','195')
dest = dest.replace('OGG','196')
dest = dest.replace('OKC','197')
dest = dest.replace('OMA','198')
dest = dest.replace('OME','199')
dest = dest.replace('ONT','200')
dest = dest.replace('ORD','201')
dest = dest.replace('ORF','202')
dest = dest.replace('OTZ','203')
dest = dest.replace('OXR','204')
dest = dest.replace('PBI','205')
dest = dest.replace('PDX','206')
dest = dest.replace('PFN','207')
dest = dest.replace('PHF','208')
dest = dest.replace('PHL','209')
dest = dest.replace('PIA','210')
dest = dest.replace('PIH','211')
dest = dest.replace('PLN','212')
dest = dest.replace('PNS','213')
dest = dest.replace('PSC','214')
dest = dest.replace('PSE','215')
dest = dest.replace('PSG','216')
dest = dest.replace('PSP','217')
dest = dest.replace('PVD','218')
dest = dest.replace('PWM','219')
dest = dest.replace('RAP','220')
dest = dest.replace('RDD','221')
dest = dest.replace('RDM','222')
dest = dest.replace('RDU','223')
dest = dest.replace('RFD','224')
dest = dest.replace('RHI','225')
dest = dest.replace('RIC','226')
dest = dest.replace('RNO','227')
dest = dest.replace('ROA','228')
dest = dest.replace('ROC','229')
dest = dest.replace('RST','230')
dest = dest.replace('RSW','231')
dest = dest.replace('SAN','232')
dest = dest.replace('SAT','233')
dest = dest.replace('SAV','234')
dest = dest.replace('SBA','235')
dest = dest.replace('SBN','236')
dest = dest.replace('SBP','237')
dest = dest.replace('SCC','238')
dest = dest.replace('SCE','239')
dest = dest.replace('SDF','240')
dest = dest.replace('SEA','241')
dest = dest.replace('SFO','242')
dest = dest.replace('SGF','243')
dest = dest.replace('SGU','244')
dest = dest.replace('SHV','245')
dest = dest.replace('SIT','246')
dest = dest.replace('SJC','247')
dest = dest.replace('SJT','248')
dest = dest.replace('SJU','249')
dest = dest.replace('SLC','250')
dest = dest.replace('SMF','251')
dest = dest.replace('SMX','252')
dest = dest.replace('SNA','253')
dest = dest.replace('SPI','254')
dest = dest.replace('SPS','255')
dest = dest.replace('SRQ','256')
dest = dest.replace('STL','257')
dest = dest.replace('STT','258')
dest = dest.replace('STX','259')
dest = dest.replace('SUN','260')
dest = dest.replace('SUX','261')
dest = dest.replace('SWF','262')
dest = dest.replace('SYR','263')
dest = dest.replace('TEX','264')
dest = dest.replace('TLH','265')
dest = dest.replace('TOL','266')
dest = dest.replace('TPA','267')
dest = dest.replace('TRI','268')
dest = dest.replace('TTN','269')
dest = dest.replace('TUL','270')
dest = dest.replace('TUP','271')
dest = dest.replace('TUS','272')
dest = dest.replace('TVC','273')
dest = dest.replace('TWF','274')
dest = dest.replace('TXK','275')
dest = dest.replace('TYR','276')
dest = dest.replace('TYS','277')
dest = dest.replace('VLD','278')
dest = dest.replace('VPS','279')
dest = dest.replace('WRG','280')
dest = dest.replace('XNA','281')
dest = dest.replace('YAK','282')
dest = dest.replace('YUM','283')
dest = dest.replace('ACK','284')
dest = dest.replace('APF','285')
dest = dest.replace('BJI','286')
dest = dest.replace('PHX','287')
dest = dest.replace('CDC','288')
dest = dest.replace('EWN','289')
dest = dest.replace('COS','290')
dest = dest.replace('GCC','291')
dest = dest.replace('GST','292')
dest = dest.replace('HHH','293')
dest = dest.replace('INL','294')
dest = dest.replace('ITH','295')
dest = dest.replace('LAW','296')
dest = dest.replace('LMT','297')
dest = dest.replace('OTH','298')
dest = dest.replace('PIR','299')
dest = dest.replace('PIT','300')
dest = dest.replace('PMD','301')
dest = dest.replace('PUB','302')
dest = dest.replace('RKS','303')
dest = dest.replace('ROW','304')
dest = dest.replace('SLE','305')
dest = dest.replace('WYS','306')
dest = dest.replace('YKM','307')
dest = dest.replace('BFF','308')
dest = dest.replace('CDC','309')
dest = dest.replace('OGD','310')
dest = dest.replace('PVU','311')
dest = dest.replace('MBS','312')
dest = dest.replace('CYS','313')
dest = dest.replace('MKG','314')
dest = dest.replace('LWB','315')

data['dest_int'] = dest

#return data

#featureeng(data.cleaning)


# In[6]:


#def featureselection():

data = data[['Month','DayofMonth','DayOfWeek','uniquecarrier_int','Cancelled','Diverted','CarrierDelay','WeatherDelay',
         'NASDelay','SecurityDelay','LateAircraftDelay','FlightNum','ArrDelay','DepDelay','origin_int','dest_int']]

df_inputs = data[['Month','DayofMonth','DayOfWeek','uniquecarrier_int','origin_int','dest_int']]

logging.info("Defining the inputs and the targets of our prediction")
df_target_reg = data[['ArrDelay','DepDelay']]

df_target_classi_1 = data[['Cancelled']]
df_target_reg_2 = data[['CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay']] 


# In[7]:


df_inputs_train ,df_inputs_test , df_target_train ,  df_target_test = train_test_split(df_inputs, df_target_classi_1, test_size=0.30,random_state=1)


gnb = GaussianNB()
logging.info("Using Naive Bayes classifier for calculating whether there is delay or not")
# Train our classifier
gnb.fit(df_inputs_train,  df_target_train)

pickle.dump(gnb,open('delay.pkl', "wb" ))
logging.info("Creating a pickel file of Naive Bayes classifier ")
#from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(max_depth=2, random_state=0) 
#clf.fit(df_inputs, df_target_classi_1) 


# In[8]:


g = gnb.predict(df_inputs_test)
from sklearn.metrics import accuracy_score
r = accuracy_score(df_target_test,g)
logging.info("Calculating the accuracy for NB classifier")
r


# In[9]:


c = confusion_matrix(df_target_test,g)
c


# In[10]:


#def model():

inputs_train ,inputs_test , target_train ,target_test = train_test_split(df_inputs, df_target_reg, test_size=0.30,random_state=1)


# In[23]:


rfr = RandomForestRegressor(n_estimators =10, random_state = 1,n_jobs=-1)
logging.info("Using Random Forest Regressor for calculating the avg departure or arrival delay value")
rfr.fit(inputs_train,target_train)
pickle.dump(rfr,open('delay_value.pkl', "wb" ))
logging.info("Creating a pickel file of Random Forest Regressor ")


# In[11]:


#from sklearn.ensemble import RandomForestClassifier 
#x_train , x_test , y_train , y_test = train_test_split(df_inputs, df_target_classi_2, test_size=0.30,random_state=1)
#clf = RandomForestClassifier(max_depth=2)
#logging.info("Using Random Forest classifier for calculating which category of delay it is")
#clf.fit(x_train,y_train)
#pickle.dump(clf,open('delay_type.pkl', "wb" ))
#logging.info("Creating a pickel file of Random Forest classifier ")
#from sklearn.ensemble import RandomForestClassifier 

x_train , x_test , y_train , y_test = train_test_split(df_inputs, df_target_reg_2, test_size=0.30,random_state=1)
rfr_2 = RandomForestRegressor(n_estimators =5, random_state = 1,n_jobs=-1)
rfr_2.fit(x_train,y_train)
pickle.dump(rfr_2,open('delay_type_value.pkl', "wb" ))


# In[24]:


parser = argparse.ArgumentParser(description='Please enter the values below :')
parser.add_argument("--Access_key", help = 'Enter Access_key of your aws account')
parser.add_argument("--Secret_key", help = 'Enter Secret_key of your aws account')
args = parser.parse_args()
logging.info("Pushing the pickle files on S3 to be used at the backened of the web interface and give prediction output ")
Access_key = args.Access_key
Secret_key = args.Secret_key

buck_name="team3"#enter bucket name
Input_location = 'us-west-2'
S3_client = boto3.client('s3', Input_location, aws_access_key_id= Access_key, aws_secret_access_key= Secret_key)

#if Input_location == 'us-east-1':
#        S3_client.create_bucket(
#            Bucket=buck_name
#        )
#else:
#    S3_client.create_bucket(
#            Bucket=buck_name,
#            CreateBucketConfiguration={'LocationConstraint': Input_location},
#        )

S3_client.upload_file('delay.pkl', buck_name, 'delay.pkl')
S3_client.upload_file('delay_value.pkl', buck_name, 'delay_value.pkl')
#S3_client.upload_file('delay_type.pkl', buck_name, 'delay_type.pkl')
S3_client.upload_file('delay_type_value.pkl', buck_name, 'delay_type_value.pkl')

