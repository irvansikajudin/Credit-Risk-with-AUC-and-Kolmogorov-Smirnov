#!/usr/bin/env python
# coding: utf-8

# # importing Library

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 99)


# # importing Data

# In[3]:


df = pd.read_csv('C:/Users/irvan/Downloads/loan_data_2007_2014.csv', index_col=0)
df.sample(2)


# # Exploring Data

# In[4]:


df.info()


# In[5]:


df['id'].nunique(), df['member_id'].nunique()


# Terlihat bahwa tidak ada id atau member_id yang duplikat, artinya setiap baris sudah mewakili satu individu.

# In[6]:


'''
ini adalah code untuk mengecek jumlah missig values yang ada pada dataset yang sedang diolah.
jika ada lebih dari 60% missing, maka akan di drop.
'''
print(df.shape)

datamissing = pd.DataFrame(df.isnull().sum().sort_values(ascending = False)).reset_index()
# datamissing = pd.DataFrame(df.isnull().sum()).reset_index()
datamissing['nama kolom'] = datamissing['index']
datamissing['jumlah missing values'] = datamissing[0]
datamissing['persentase'] = round(datamissing['jumlah missing values']/df.shape[0]*100,)
datamissing.drop(['index',0], axis=1, inplace=True )
# pd.set_option('display.max_rows', None) # max row
datamissing


# In[7]:


#kolom yang memiliki missing values lebih dari 60% akan di drop
#sehingga akan dimasukkan ke list
datamissing = datamissing[datamissing['persentase']>60]

#list drop ==> missing values > 60%
columnsToDrop = datamissing['nama kolom'].tolist()
columnsToDrop2 = [

#unique id
'id'
 ,'member_id'
#free text
 ,'url'
#lexpert judgment
 ,'sub_grade'                
#constant/orhers
 ,'zip_code']

columnsToDrop = columnsToDrop + columnsToDrop2


# In[8]:


df.drop(columnsToDrop, axis=1, inplace=True)


# In[9]:


df.shape


# kolom berhasil dibuang sehingga menjadi 49 kolom

# # Definition of Label
# loan status cocok untuk dijadikan label untuk machine learning. mari kita cek nilai dari loan_status.

# In[10]:


df['loan_status'].value_counts()/len(df)*100


# terlihat ada banyak kategori pada kolom loan_status

# 1. Current artinya pembayaran lancar; 
# 2. Charged Off artinya pembayaran macet sehingga dihapusbukukan; 
# 3. Late artinya pembayaran telat dilakukan; 
# 4. In Grace Period artinya dalam masa tenggang; 
# 5. Fully Paid artinya pembayaran lunas; Default artinya pembayaran macet
# 
# Dari definisi-definisi tersebut, masing-masing individu dapat ditandai apakah mereka merupakan peminjam yang buruk atau peminjam yang baik. maka dari itu akan saya definisikan sebagai bad load dan good loan.
# 
# karena saya tidak tau yang dimaksud bad load pada instansi yang memiliki data, maka dari itu saya menggunakan keterlambatan pembayaran di atas 30 hari dan yang lebih buruk dari itu sebagai penanda bad loan.

# In[11]:


bad_status = [
    'Charged Off' 
    , 'Default' 
    , 'Does not meet the credit policy. Status:Charged Off'
    , 'Late (31-120 days)'
]

df['target'] = np.where(df['loan_status'].isin(bad_status), 1, 0) # merubah semua nilai yg ada di list bad_status ke 1 ==> badloan,0 ==> goodloan
df.drop('loan_status', axis=1, inplace=True) # drop fitur asli ==> loan_status
df['target'].value_counts()/len(df)*100 # cek balance dari target

'''
1 = adalah bad loan, peminjam dengan prilaku buruk
0 = peminjam good loan, peminjam dengan prilaku baik
'''


# # EDA part 1

# ## Numerical Data

# In[12]:


numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
print('\nKita dapat mengamati bahwa data memiliki',df.shape[0],'baris dan memiliki', df.select_dtypes(include=numerics).shape[1],'kolom numerik')
# print('kolom numerik pada data ==> ', list(df.select_dtypes(include=numerics).columns),' \n')
df_num = df.select_dtypes(include=numerics)
df_num.head(3)


# In[13]:


df.select_dtypes(exclude='object').nunique()


# policy_code walaupun data numerical hanya memiliki 1 unik value, maka dari itu juga akan di drop.

# In[14]:


df.drop('policy_code', axis=1, inplace=True)


# ## Non Numerical Data

# In[15]:


df_cat = df.select_dtypes(include=object)
rows = []
for col in df_cat :
  rows.append(
        {
            'Nama Kolom': col,
            'Jumlah Unique Values': len(df[col].unique()),
            'Data Type': df[col].dtypes,
            'Unique Values':  df[col].unique()
        }
  )
pd.options.display.max_colwidth = 170 #maksimal tampil 170 karakter
# pd.options.display.max_colwidth = 70 #maksimal tampil 70 karakter
# pd.options.display.max_colwidth #tanpa batas 
print('\nInformasi Kolom Kategori : \n') 
unik = pd.DataFrame(rows)
unik.sort_values(by='Jumlah Unique Values', ascending=False).reset_index().drop('index', 1)


# kolom application_type hanya 1 value unik, maka dari itu akan di drop juga, kemudian emp_title dan title memiliki nilai unik yang sangat tinggi (high cardinality) juga akan di drop

# In[16]:


df.drop(['application_type','emp_title','title'], axis=1, inplace=True)


# # CLEANING, PREPROCESSING, FEATURE ENGINEERING part1

# kolom emp_length dan term bercampur dengan angka, maka dari itu string akan dihapus dan akan di convert menjadi numeric

# ## emp_length

# In[17]:


df['emp_length'].unique(), df['term'].unique()


# In[18]:


df['emp_length_int'] = df['emp_length'].str.replace('\+ years', '')
df['emp_length_int'] = df['emp_length_int'].str.replace('< 1 year', str(0))
df['emp_length_int'] = df['emp_length_int'].str.replace(' years', '')
df['emp_length_int'] = df['emp_length_int'].str.replace(' year', '')


# In[19]:


df['emp_length_int'] = df['emp_length_int'].astype(float)
df.drop('emp_length', axis=1, inplace=True)


# In[20]:


df['emp_length_int'].value_counts()


# ## term

# In[21]:


df['term'].unique()


# Memodifikasi term. Contoh: 36 months -> 36

# In[22]:


df['term_int'] = df['term'].str.replace(' months', '')
df['term_int'] = df['term_int'].astype(float)
df.drop('term', axis=1, inplace=True)


# ## earliest_cr_line
# earliest_cr_line adalah Bulan dimana batas kredit paling awal yang dilaporkan peminjam dibuka</br>
# Memodifikasi earliest_cr_line dari bentuk asalnya adalah bulan-tahun menjadi perhitungan berapa lama waktu berlalu sejak waktu tersebut. maka dari itu biasanya akan menggunakan reference date = hari ini. tapi, karena dataset ini adalah dataset tahun 2007-2014, maka akan lebih relevan jika menggunakan reference date di sekitar tahun 2017. maka dari itu, saya akan mencoba menggunakan tanggal 2017-12-01 sebagai reference date.

# In[23]:


df['earliest_cr_line'].head(3)


# bentuknya bulan-tahun, kita akan rubah menjadi tahun-bulan-tanggal

# In[24]:


'''
funsi days_of_future ada untuk menghindari python salah dalam menconvert tanggal,
contoh : Apr-99 menjadi 2099-04-01
sehingga python dapat menkonversi tgl sesuai dgn tanggal sebenanya,
contoh : Apr-99 menjadi 1999-04-01
'''

def days_of_future_past(date,chk_y=pd.Timestamp.today().year):
    return date.replace(year=date.year-100) if date.year > chk_y else date
 

df['earliest_cr_line_tahunbulanhari'] = pd.to_datetime(df['earliest_cr_line'],format='%b-%y').map(days_of_future_past)
df['earliest_cr_line_tahunbulanhari'].head(3)


# kemudian akan dikurangin dengan 2017-12-01 untuk mengetahui seberapa lama telah berlalu dari earliest_cr_line/earliest_cr_line_tahunbulanhari

# In[25]:


df['bulan_berlalu_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - 
                                                        df['earliest_cr_line_tahunbulanhari']) / 
                                                        np.timedelta64(1, 'M')))
'''
2017-12-01 dikurang dengan 'earliest_cr_line_tahunbulanhari'
kumdian dibagi bulan untuk mengetahui berapa bulan telah
berlalu.
'''
print('hasil perhitungan untuk mengetahui sudah berapa bulan berlalu :')
display(df['bulan_berlalu_since_earliest_cr_line'].head(5))

print('\nDescribe :')
display(df['bulan_berlalu_since_earliest_cr_line'].describe())


# In[26]:


'''
kode untuk mengecek apakah ada nilai minus/salah konversi tgl
'''

df[df['bulan_berlalu_since_earliest_cr_line']<0][['earliest_cr_line', 'earliest_cr_line_tahunbulanhari', 'bulan_berlalu_since_earliest_cr_line']].head()


# sepertinya aman, lanjut saja

# In[27]:


df.drop(['earliest_cr_line', 'earliest_cr_line_tahunbulanhari'], axis=1, inplace=True)


# ## issue_d

# prosesnya sama dgn earliest_cr_line

# In[28]:


df['issue_d_tahunbulanhari'] = pd.to_datetime(df['issue_d'],format='%b-%y').map(days_of_future_past)
df['bulan_berlalu_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - df['issue_d_tahunbulanhari']) / np.timedelta64(1, 'M')))
df.drop(['issue_d', 'issue_d_tahunbulanhari'], axis=1, inplace=True)
df['bulan_berlalu_since_issue_d'].describe()


# ## last_pymnt_d

# prosesnya sama dgn earliest_cr_line

# In[29]:


df['last_pymnt_d_tahunbulanhari'] = pd.to_datetime(df['last_pymnt_d'],format='%b-%y').map(days_of_future_past)
df['bulan_berlalu_since_last_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - df['last_pymnt_d_tahunbulanhari']) / np.timedelta64(1, 'M')))
df.drop(['last_pymnt_d', 'last_pymnt_d_tahunbulanhari'], axis=1, inplace=True)
df['bulan_berlalu_since_issue_d'].describe()


# ## next_pymnt_d

# prosesnya sama dgn earliest_cr_line

# In[30]:


df['next_pymnt_d_tahunbulanhari'] = pd.to_datetime(df['next_pymnt_d'],format='%b-%y').map(days_of_future_past)
df['bulan_berlalu_since_next_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - df['next_pymnt_d_tahunbulanhari']) / np.timedelta64(1, 'M')))
df.drop(['next_pymnt_d', 'next_pymnt_d_tahunbulanhari'], axis=1, inplace=True)
df['bulan_berlalu_since_next_pymnt_d'].describe()


# ## last_credit_pull_d

# prosesnya sama dgn earliest_cr_line

# In[31]:


df['last_credit_pull_d_tahunbulanhari'] = pd.to_datetime(df['last_credit_pull_d'],format='%b-%y').map(days_of_future_past)
df['bulan_berlalu_since_last_credit_pull_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - df['last_credit_pull_d_tahunbulanhari']) / np.timedelta64(1, 'M')))
df.drop(['last_credit_pull_d', 'last_credit_pull_d_tahunbulanhari'], axis=1, inplace=True)
df['bulan_berlalu_since_last_credit_pull_d'].describe()


# # EDA part 2

# ## Correlation check

# In[32]:


plt.figure(figsize=(30,30))
sns.heatmap(df.corr(), annot=True, fmt = ".2f", cmap = "BuPu")


# jika ada pasangan fitur-fitur yang memiliki korelasi tinggi maka akan diambil salah satu saja. Nilai korelasi yang dijadikan patokan sebagai korelasi tinggi tidak pasti, biasanya pada angka 0.7 namun angka ini tergantung data scientist itu sendiri

# In[33]:


corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop_hicorr = [column for column in upper.columns if any(upper[column] > 0.7)]
to_drop_hicorr


# In[34]:


df.drop(to_drop_hicorr, axis=1, inplace=True)


# ## Pairplot

# In[35]:


# plt.figure(figsize=(160,100))
# sns.pairplot(df,
#             #  hue='Turbo',
#              corner=True)


# pariplot tidak cocok dengan data dengan fitur banyak

# # CLEANING, PREPROCESSING, FEATURE ENGINEERING part2

# ## Missing Values

# In[36]:


'''
ini adalah code untuk mengecek jumlah missig values yang ada pada dataset yang sedang diolah.
jika ada lebih dari 60% missing, maka akan di drop.
'''
print('total baris', df.shape[0], 'dan total kolom', df.shape[1])
print('total kolom yg memiliki missing values =', datamissing[datamissing['persentase']>0].shape[0])

datamissing = pd.DataFrame(df.isnull().sum().sort_values(ascending = False)).reset_index()
# datamissing = pd.DataFrame(df.isnull().sum()).reset_index()
datamissing['nama kolom'] = datamissing['index']
datamissing['jumlah missing values'] = datamissing[0]
datamissing['persentase'] = datamissing['jumlah missing values']/df.shape[0]*100
datamissing.drop(['index',0], axis=1, inplace=True )
# pd.set_option('display.max_rows', None) # max row
datamissing[datamissing['persentase']>0]


# setiap fitur yg memiliki missing values tidak mencapai 5% maka baris yg memiliki missing values akan langsung di hapus saja

# In[37]:


dropcol = datamissing[(datamissing['persentase']>0) & (datamissing['persentase']<5)]
dropcol = dropcol['nama kolom'].tolist()
dropcol


# In[38]:


df.dropna(subset=dropcol, inplace=True)


# In[39]:


df.shape


# In[40]:


'''
ini adalah code untuk mengecek jumlah missig values yang ada pada dataset yang sedang diolah.
jika ada lebih dari 60% missing, maka akan di drop.
'''
print('total baris', df.shape[0], 'dan total kolom', df.shape[1])


datamissing = pd.DataFrame(df.isnull().sum().sort_values(ascending = False)).reset_index()
# datamissing = pd.DataFrame(df.isnull().sum()).reset_index()
datamissing['nama kolom'] = datamissing['index']
datamissing['jumlah missing values'] = datamissing[0]
datamissing['persentase'] = datamissing['jumlah missing values']/df.shape[0]*100
datamissing.drop(['index',0], axis=1, inplace=True )
# pd.set_option('display.max_rows', None) # max row
display(datamissing[datamissing['persentase']>0])
print('\n\ntotal kolom yg memiliki missing values =', datamissing[datamissing['persentase']>0].shape[0])


# ### Missing Values Filling

# In[41]:


df['tot_cur_bal'].fillna(0, inplace=True)
df['tot_coll_amt'].fillna(0, inplace=True)
df['mths_since_last_delinq'].fillna(-1, inplace=True)


# In[42]:


'''
ini adalah code untuk mengecek jumlah missig values yang ada pada dataset yang sedang diolah.
jika ada lebih dari 60% missing, maka akan di drop.
'''
print('total baris', df.shape[0], 'dan total kolom', df.shape[1])


datamissing = pd.DataFrame(df.isnull().sum().sort_values(ascending = False)).reset_index()
# datamissing = pd.DataFrame(df.isnull().sum()).reset_index()
datamissing['nama kolom'] = datamissing['index']
datamissing['jumlah missing values'] = datamissing[0]
datamissing['persentase'] = datamissing['jumlah missing values']/df.shape[0]*100
datamissing.drop(['index',0], axis=1, inplace=True )
# pd.set_option('display.max_rows', None) # max row
display(datamissing[datamissing['persentase']>0])
print('\n\ntotal kolom yg memiliki missing values =', datamissing[datamissing['persentase']>0].shape[0])


# ## Duplicate Values

# In[43]:


dup = df.duplicated().sum()/df.shape[0]*100
print('Persentase df duplicate dari keseluruhan df adalah ',round(dup,2), 'persen, dengan total baris yang duplicate adalah',df.duplicated().sum() )


# # EDA part 3

# In[44]:


dfeda = df.copy()


# In[84]:


dfeda_cat = dfeda.select_dtypes(include='object').columns.tolist()
dfeda_num = dfeda.select_dtypes(exclude='object').columns.tolist()

len(dfeda_cat), len(dfeda_num)


# In[123]:


dfeda_cat


# In[89]:


numericals = dfeda_num

plt.figure(figsize=(15, 10))
for i in range(0, len(numericals)):
    plt.subplot(7, 4, i+1) # 7,2 maksudnya 7x2=14 kolom, sesuaikan dengan jumlah kolom numerik
    sns.boxplot(x=dfeda[numericals[i]], color='orange')
    plt.tight_layout()

plt.show()


# In[156]:


numericals = dfeda_num

plt.figure(figsize=(15, 10))
for i in range(0, len(numericals)):
    plt.subplot(7, 4, i+1) # 7,2 maksudnya 7x2=14 kolom, sesuaikan dengan jumlah kolom numerik
    sns.distplot(dfeda[numericals[i]], color='orange')
    plt.tight_layout()

plt.show()


# In[99]:


numericals = dfeda_num

plt.figure(figsize=(14, 35))
for i in range(0, len(numericals)):
    plt.subplot(7, 6, i+1) # 7,6 demensi
    sns.boxplot(x="target", y=boxplot_catplot[i],data=dfeda)
    plt.tight_layout()

plt.show()


# In[154]:


numericals = dfeda_cat

plt.figure(figsize=(14, 35))
for i in range(0, len(numericals)):
    plt.subplot(7, 3, i+1) # 7,6 demensi
    sns.countplot(x=dfeda_cat[i],  data=dfeda , palette='Paired',order=pd.value_counts(dfeda[dfeda_cat[i]]).iloc[:5].index)
    plt.xlabel("")
    plt.tick_params(axis='x', rotation=90)
    plt.title('5 Top Teratas dari ' + dfeda_cat[i])
    plt.tight_layout()

plt.show()


# saya malas untuk membuat interpretasi dari bagan-bagan di atas, saya serahkan interpretasinya kepada kemampuan anda, hehe  maap

# # FEATURE SCALING AND TRANSFORMATION

# ## categorical values check

# In[45]:


colcat = pd.DataFrame(df.select_dtypes(include='object').nunique()).reset_index()
colcat['nama kolom'] = colcat['index']
colcat['unique'] = colcat[0]
colcat.drop(['index',0], axis=1, inplace=True)
colcat


# ## one hot encoding

# In[46]:


colcat = colcat['nama kolom'].tolist()
onehot = pd.get_dummies(df[colcat], drop_first=True)
onehot.head()


# ## Standardization

# Semua kolom numerikal dilakukan proses standarisasi dengan StandardScaler (fitur target dikecualikan)

# In[47]:


numerical_cols = [col for col in df.columns.tolist() if col not in colcat + ['target']]


# In[48]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
std = pd.DataFrame(ss.fit_transform(df[numerical_cols]), columns=numerical_cols)


# In[49]:


std.head()


# ## Transformed Dataframe

# Menggabungkan kembali kolom-kolom hasil transformasi, sebelum menggabungkan alangkah lebih baik melakukan reset index, untuk menghindari pandas menghasilkan value NaN gara-gara index yang tidak sama

# In[50]:


onehot.reset_index(drop=True, inplace=True)
std.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)


# In[51]:


data_model = pd.concat([onehot, std, df[['target']]], axis=1)
data_model.shape,onehot.shape,std.shape,df.shape


# # Import evaluation Library

# In[52]:


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


# # Hyperparameter Tuning XgBoost with Hyperopt
# Link : https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook </br>
# The available hyperopt optimization algorithms are -
# 
# hp.choice(label, options) — Returns one of the options, which should be a list or tuple.
# 
# hp.randint(label, upper) — Returns a random integer between the range [0, upper).
# 
# hp.uniform(label, low, high) — Returns a value uniformly between low and high.
# 
# hp.quniform(label, low, high, q) — Returns a value round(uniform(low, high) / q) * q, i.e it rounds the decimal values and returns an integer.
# 
# hp.normal(label, mean, std) — Returns a real value that’s normally-distributed with mean and standard deviation sigma.

# In[53]:


# !pip install hyperopt


# In[53]:


X = data_model.drop('target', axis=1)
y = data_model['target']


# In[54]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,
                                                y,
                                                test_size = 0.3,
                                                random_state = 42)


# ## Using Hyperopt

# In[56]:


# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }


# In[57]:


def objective(space):
    clf=xgb.XGBClassifier(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }


# In[58]:


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[59]:


trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)


# In[60]:


print("The best hyperparameters are : ","\n")
print(best_hyperparams)


# In[61]:


# convert to string
def hapus_karakter_unik(a):
    a = a.replace("'","") 
    a = a.replace('"',"'")
    a = a.replace('{',"")
    a = a.replace('}',"")
    a = a.replace(':',"=")
    a = a.replace('.0,',",")
    return a

best_hyperparameters = str(best_hyperparams)
best_hyperparameters = hapus_karakter_unik(best_hyperparameters)
print(best_hyperparameters)
# def paramTuning():
#     return print(best_hyperparameters)
# paramTuning()


# In[62]:


from hyperopt import space_eval
bparamsdict = space_eval(space, best_hyperparams)
xgb_best_params_tuning = pd.DataFrame(list(bparamsdict.items()), columns = ['parameter','Value'])
bestparams0 = float(xgb_best_params_tuning['Value'][0])
bestparams1 = float(xgb_best_params_tuning['Value'][1]) 
bestparams2 = int(xgb_best_params_tuning['Value'][2]) 
bestparams3 = int(xgb_best_params_tuning['Value'][3])
bestparams4 = int(xgb_best_params_tuning['Value'][4])
bestparams5 = int(xgb_best_params_tuning['Value'][5]) 
bestparams6 = float(xgb_best_params_tuning['Value'][6])
bestparams7 = int(xgb_best_params_tuning['Value'][7])
xgb_best_params_tuning


# ## Evaluasi dengan parameter terbaik yang ditemukan pada proses hyperparameter tuning seblumnya.

# In[158]:


X = data_model.drop('target', axis=1)
y = data_model['target']

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,
                                                y,
                                                test_size = 0.3,
                                                random_state = 42)


# In[60]:


# Fit - train
import xgboost as xgb

# Penyetingan menggunakan parameter terbaik.
xgb = xgb.XGBClassifier(colsample_bytree = bestparams0, 
                        gamma = bestparams1, 
                        max_depth = bestparams2, 
                        min_child_weight = bestparams3,
#                         n_estimators = bestparams4,
                        reg_alpha = bestparams5, 
#                         seed = bestparams7,
                       reg_lambda = bestparams6)

# Penyetingan parameter secara manual
# xgb = xgb.XGBClassifier(colsample_bytree= 0.7508546033482361, 
#                         gamma= 1.8810689409604566, 
#                         max_depth= 14, 
#                         min_child_weight= 9, 
#                         reg_alpha= 109, 
#                         reg_lambda= 0.4775669801060372)

# Train
xgb.fit(X_train, y_train)

# Predict
y_pred_xgb_tuning_train = xgb.predict(X_train) # prediksi train dataset
y_pred_xgb_tuning_test = xgb.predict(X_test) # prediksi test dataset


#EValuasi
accuracy_xgb_tuning_train = accuracy_score(y_train, y_pred_xgb_tuning_train)
accuracy_xgb_tuning_test = accuracy_score(y_test, y_pred_xgb_tuning_test)

precision_xgb_tuning_train = precision_score(y_train, y_pred_xgb_tuning_train)
precision_xgb_tuning_test = precision_score(y_test, y_pred_xgb_tuning_test)

recall_xgb_tuning_train = recall_score(y_train, y_pred_xgb_tuning_train)
recall_xgb_tuning_test = recall_score(y_test, y_pred_xgb_tuning_test)

specificity_xgb_tuning_train = recall_score(y_train, y_pred_xgb_tuning_train, pos_label=0)
specificity_xgb_tuning_test = recall_score(y_test, y_pred_xgb_tuning_test, pos_label=0)

F1_xgb_tuning_train = f1_score(y_train, y_pred_xgb_tuning_train)
F1_xgb_tuning_test = f1_score(y_test, y_pred_xgb_tuning_test)

roc_auc_xgb_tuning_train = roc_auc_score(y_train, y_pred_xgb_tuning_train)
roc_auc_xgb_tuning_test = roc_auc_score(y_test, y_pred_xgb_tuning_test)

print('Accuracy Train - Tuning\t\t: %.2f' % accuracy_xgb_tuning_train)
print('Accuracy Test - Tuning\t\t: %.2f' % accuracy_xgb_tuning_test)

print('Precision Train - Tuning\t: %.2f' % precision_xgb_tuning_train)
print('Precision Test - Tuning\t\t: %.2f' % precision_xgb_tuning_test)

print('Recall Train - Tuning\t\t: %.2f' % recall_xgb_tuning_train)
print('Recall Test - Tuning\t\t: %.2f' % recall_xgb_tuning_test)

print('specificity Train - Tuning\t: %.2f' % specificity_xgb_tuning_train)
print('specificity Test - Tuning\t: %.2f' % specificity_xgb_tuning_test)

print('F1 Score Train - Tuning\t\t: %.2f' % F1_xgb_tuning_train)
print('F1 Score Test - Tuning\t\t: %.2f' % F1_xgb_tuning_test)

print('Roc-AUC Train - Tuning \t\t: %.2f' % roc_auc_xgb_tuning_train)
print('Roc-AUC Test - Tuning \t\t: %.2f' % roc_auc_xgb_tuning_test)

from sklearn.metrics import classification_report, confusion_matrix
print('\n Report Klasifikasi Train :')
print('-----------------------------------------------------------\n')
print(classification_report(y_train,y_pred_xgb_tuning_train))

print('\n Report Klasifikasi Test :')
print('-----------------------------------------------------------\n')
print(classification_report(y_test,y_pred_xgb_tuning_test))


#Get the confusion matrix
cf_matrix = confusion_matrix(y_test,y_pred_xgb_tuning_test)


# Visualization Confusion Matrix on test dataset
print('\n Confusion Matrix Test :')
print('-----------------------------------------------------------\n')
group_names = ['True Negatif','False Positif','False Negatif','True Positif']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
plt.figure(figsize = (7,3))
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


# # Hyperparameter Tuning  Random Forest with GridseacrhCV

# In[65]:


X = data_model.drop('target', axis=1)
y = data_model['target']

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,
                                                y,
                                                test_size = 0.3,
                                                random_state = 42)


# In[66]:


# define random forest classifier model
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=42)


# ## Using GridsearchCV

# In[67]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import GridSearchCV\n\nparameters = {\n    'n_estimators': (10,20,30,40,50),\n    # 'max_depth':(1,2,3,4,5)\n    'max_depth':list(range(10, 15))\n\n}\n\n# note: we use metrik yg telah ditentukan sesuai tujuan (recall/akurasi/presisi/Dll)\nrf_clf_gridcv = GridSearchCV(rf_clf, parameters, cv=5, scoring='accuracy') \nrf_clf_gridcv.fit(X_train, y_train)\n\n# the results\ncv_result = pd.DataFrame(rf_clf_gridcv.cv_results_)\nretain_cols = ['params','mean_test_score','rank_test_score']\ncv_result[retain_cols].sort_values('rank_test_score')")


# In[68]:


import pandas as pd
import matplotlib.pyplot as plt

# creating a DataFrame with 2 columns
a = cv_result['mean_test_score'].tolist()
dataFrame = pd.DataFrame(
   {
      'mean_test_score': a
   }
)

# plot a line graph
plt.plot(dataFrame["mean_test_score"])
# plt.ylim(160, 180)
plt.rcParams["figure.figsize"] = (20,4)
plt.rcParams['axes.titlesize'] = (13)
# displaying the title
plt.title("Mean Test Score of Grid Search Cross Validation Result with K-Fold Validation = 5")
plt.ylabel("Mean Test Score", fontsize=12)
plt.xlabel("25 kali Cross Validation yg dilakukan - 5 kali training setiap  Cross Validation", fontsize=12)
# plt.ylim(0, 1)
plt.show()


# ## Evaluasi dengan parameter terbaik yang ditemukan pada proses hyperparameter tuning seblumnya.

# In[61]:


X = data_model.drop('target', axis=1)
y = data_model['target']

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,
                                                y,
                                                test_size = 0.3,
                                                random_state = 42)


# In[70]:


rf_clf_gridcv.best_estimator_


# In[62]:


# Fit - train
from sklearn.ensemble import RandomForestClassifier
# rf = rf_clf_randomcv.best_estimator_ # best parameter yang dihasilkan randomizedsearchcv
rf = rf_clf_gridcv.best_estimator_ # best parameter yang dihasilkan gridsearchcv
# rf = RandomForestClassifier(random_state=42,n_estimators=30, max_depth=5) # jika ingin menggunakan parameter yang diset secara manual

# Train
rf.fit(X_train,y_train)

# Predict
y_pred_rf_tuning_train = rf.predict(X_train) # prediksi train dataset
y_pred_rf_tuning_test = rf.predict(X_test) # prediksi test dataset

#EValuasi
accuracy_tuning_rf_tuning_train = accuracy_score(y_train, y_pred_rf_tuning_train)
accuracy_tuning_rf_tuning_test = accuracy_score(y_test, y_pred_rf_tuning_test)

precision_tuning_rf_tuning_train = precision_score(y_train, y_pred_rf_tuning_train)
precision_tuning_rf_tuning_test = precision_score(y_test, y_pred_rf_tuning_test)

recall_tuning_rf_tuning_train = recall_score(y_train, y_pred_rf_tuning_train)
recall_tuning_rf_tuning_test = recall_score(y_test, y_pred_rf_tuning_test)

specificity_tuning_rf_tuning_train = recall_score(y_train, y_pred_rf_tuning_train, pos_label=0)
specificity_tuning_rf_tuning_test = recall_score(y_test, y_pred_rf_tuning_test, pos_label=0)

F1_tuning_rf_tuning_train = f1_score(y_train, y_pred_rf_tuning_train)
F1_tuning_rf_tuning_test = f1_score(y_test, y_pred_rf_tuning_test)

roc_auc_rf_tuning_train = roc_auc_score(y_train, y_pred_rf_tuning_train)
roc_auc_rf_tuning_test = roc_auc_score(y_test, y_pred_rf_tuning_test)

print('Accuracy Train\t\t: %.2f' % accuracy_tuning_rf_tuning_train)
print('Accuracy Test\t\t: %.2f' % accuracy_tuning_rf_tuning_test)

print('Precision Train\t\t: %.2f' % precision_tuning_rf_tuning_train)
print('Precision Test\t\t: %.2f' % precision_tuning_rf_tuning_test)

print('Recall Train\t\t: %.2f' % recall_tuning_rf_tuning_train)
print('Recall Test\t\t: %.2f' % recall_tuning_rf_tuning_test)

print('specificity Train\t: %.2f' % specificity_tuning_rf_tuning_train)
print('specificity Test\t: %.2f' % specificity_tuning_rf_tuning_test)

print('F1 Score Train\t\t: %.2f' % F1_tuning_rf_tuning_train)
print('F1 Score Test\t\t: %.2f' % F1_tuning_rf_tuning_test)

print('Roc-AUC Train \t\t: %.2f' % roc_auc_rf_tuning_train)
print('Roc-AUC Test \t\t: %.2f' % roc_auc_rf_tuning_test)

from sklearn.metrics import classification_report, confusion_matrix
print('\n Report Klasifikasi Train :')
print('-----------------------------------------------------------\n')
print(classification_report(y_train,y_pred_rf_tuning_train))

print('\n Report Klasifikasi Test :')
print('-----------------------------------------------------------\n')
print(classification_report(y_test,y_pred_rf_tuning_test))


#Get the confusion matrix
cf_matrix = confusion_matrix(y_test,y_pred_rf_tuning_test)


# Visualization Confusion Matrix on test dataset
print('\n Confusion Matrix Test :')
print('-----------------------------------------------------------\n')
group_names = ['True Negatif','False Positif','False Negatif','True Positif']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
plt.figure(figsize = (7,3))
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


# # KS - RFC

# In[63]:


y_pred_proba = rf.predict_proba(X_test)[:][:,1]


# In[64]:


df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_test.index


# In[65]:


df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[66]:


df_actual_predicted.head(3)


# In[67]:


KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='g')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='r')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[68]:


ks_rfc_test = KS


# In[69]:


y_pred_proba = rf.predict_proba(X_train)[:][:,1]


# In[70]:


df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_train), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_train.index


# In[71]:


df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[72]:


df_actual_predicted.head(3)


# In[73]:


KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='g')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='r')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[74]:


ks_rfc_train = KS


# # KS - XGB

# In[75]:


y_pred_proba = xgb.predict_proba(X_test)[:][:,1]


# In[76]:


df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_test.index


# In[77]:


df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[78]:


df_actual_predicted.head(3)


# In[79]:


KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='g')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='r')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[80]:


ks_xgb_test = KS


# In[81]:


y_pred_proba = xgb.predict_proba(X_train)[:][:,1]


# In[82]:


df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_train), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_train.index


# In[83]:


df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[84]:


df_actual_predicted.head(3)


# In[85]:


KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='g')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='r')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[86]:


ks_xgb_train = KS


# # Evaluasi hasil dari Hyperparame tuning

# In[87]:


result = pd.DataFrame({'Algorithm used':['XgBoost - Tuning','Random Forest - Tuning'],
                        'Accuracy Train':[accuracy_xgb_tuning_train,accuracy_tuning_rf_tuning_train],
                       'Accuracy Test':[accuracy_xgb_tuning_test,accuracy_tuning_rf_tuning_test],
#                         'Recall Train':[recall_xgb_tuning_train,recall_tuning_rf_tuning_train],
#                        'Recall Test':[recall_xgb_tuning_test,recall_tuning_rf_tuning_test],
                        'Presisi Train':[precision_xgb_tuning_train,precision_tuning_rf_tuning_train],
                        'Presisi Test':[precision_xgb_tuning_test,precision_tuning_rf_tuning_test],
                        'specificity Train':[specificity_xgb_tuning_train,specificity_tuning_rf_tuning_train],
                        'specificity Test':[specificity_xgb_tuning_test,specificity_tuning_rf_tuning_test],
#                         'F1 Score Train':[F1_xgb_tuning_train,F1_tuning_rf_tuning_train],
#                         'F1 Score Test':[F1_xgb_tuning_test,F1_tuning_rf_tuning_test],
                       'ROC-AUC Train':[roc_auc_xgb_tuning_train,roc_auc_rf_tuning_train],
                       'Roc-AUC Test':[roc_auc_xgb_tuning_test,roc_auc_rf_tuning_test],
                       'KS-Train':[ks_xgb_train,ks_rfc_train],
                       'KS-Test':[ks_xgb_test,ks_rfc_test]
                        })
# result
sorted_df = result.sort_values(by=['Accuracy Test'], ascending=False).reset_index().drop('index', 1)
sorted_df.round(2)


# # Feature Importance

# ## Feature Importance

# In[69]:


# Random Forest

arr_feature_importances = rf.feature_importances_
arr_feature_names = X_train.columns.values
    
df_feature_importance = pd.DataFrame(index=range(len(arr_feature_importances)), columns=['feature', 'importance'])
df_feature_importance['feature'] = arr_feature_names
df_feature_importance['importance'] = arr_feature_importances
df_all_features = df_feature_importance.sort_values(by='importance', ascending=False)
df_all_features.head(5)


# In[72]:


# XgBoost

arr_feature_importances = xgb.feature_importances_
arr_feature_names = X_train.columns.values
    
df_feature_importance = pd.DataFrame(index=range(len(arr_feature_importances)), columns=['feature', 'importance'])
df_feature_importance['feature'] = arr_feature_names
df_feature_importance['importance'] = arr_feature_importances
df_all_features = df_feature_importance.sort_values(by='importance', ascending=False)
df_all_features.head(5)


# ## Feature Importance with Shap

# In[73]:


import shap


# In[74]:


explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)


# In[75]:


shap.summary_plot(shap_values, X_test, plot_type="bar")


# In[76]:


shap.summary_plot(shap_values, X_test)


# link : https://mljar.com/blog/feature-importance-xgboost/#:~:text=Xgboost%20is%20a%20gradient%20boosting,%2C%20R%2C%20Julia%2C%20Scala.

# # Conclusion

# Beberapa Kesimpulan yg didapatkan pada projek ini :
# - Model terbaik setelah melakukan Hyperparameter Tuning ialah XgBoost, karena value dari smua matrix paling stabil.
# - Fitur importance pada projek ini yakni Recoveries, out_prncp, int_rate, anual_income dan seterusnya.
# - karena data imbalance maka menggunakan metric AUC dan KS, kedua Matrix ini  menunjukkan nilai yg baik(AUC = 74-75, KS = 61-62), karena pada dunia credit risk modeling, umumnya AUC di atas 0.7 dan KS di atas 0.3 sudah termasuk performa yang baik, silahkan lihat bagian "evaluasi hasil dari Hyperparameter tuning" untuk informasi lebih lanjut.
# - ![image.png](attachment:image.png)
# - Tidak melakukan SMOTE (Metode untuk membuat target Balance).
# - Jika menginginkan interpretabilitas yang lebih tinggi, dapat mempertimbangkan untuk membuat Credit Scorecard dengan menggunakan algoritma Logistic Regression dengan pendekatan-pendekatannya seperti Feature Selection menggunakan Information Value dan Feature Engineering menggunakan Weight of Evidence.
# - Menggunakan Library Hyperopt untuk tuning algoritma Xgboost dan GridSearchCV untuk tuning algoritma RandomForest, namun sebaiknya menggunakan RandomsearchCV daripada GridSearchCV agar waktu hyperparametertuning dapat lebih singkat.
# - Model tidak memiliki Overfitting/underfitting, dapat dilihat pada bagian "evaluasi hasil dari Hyperparameter tuning" untuk informasi lebih lanjut.
# - 74 Fitur pada Raw Dataset yang kemudian di proses dgn mendrop beberapa fitur dan diteruskan dengan encoding sehingga fitur menjadi 101 firtur
# - kesimpulan terakhir jika kita menginginkan interpretabilitas yang lebih tinggi, dapat mempertimbangkan untuk membuat Credit Scorecard dengan menggunakan algoritma Logistic Regression dengan melakukan Feature Selection menggunakan Information Value dan Feature Engineering menggunakan Weight of Evidence.
# - kesimpulan dari EDA silahkan interpretasi sendiri.
