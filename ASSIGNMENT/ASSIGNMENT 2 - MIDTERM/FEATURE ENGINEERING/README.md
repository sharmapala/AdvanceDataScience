# FEATURE ENGINEERING THAT WE USED :

<b>After performing Exploratory Data Analysis on our dataset we have a picture about
the overall data. We know what variables occur most , at what time , what
variables may be important and what variables may influence the target which is
the appliances.<br><br>
Once that is done , it is important to check the dataset again for outliers and
missing data or NaN values before we start the modeling.
So , for that feature engineering is performed on the dataset.<b>

<ol>
<li><b>Log Transform :<b>
<li><b>Scaling the outliers :<b>
<ol>
<li><b>STANDARD SCALER :<b>
Here , it calculates the mean and standard deviation of a sample and when it scales
the next sample it uses this stored value to scale that sample.
<li><b>MINMAX SCALER :<b>
This is similar to normalized scaler which scales the data in the range [0,1] so any
high value will be equal to 1 or close to 1.
<li><b>MAXABS SCALER :<b>
MaxAbsScaler differs from the previous scaler such that the absolute values are
mapped in the range [0, 1]. On positive only data, this scaler behaves similarly to
MinMaxScaler and therefore also suffers from the presence of large outliers.
<li><b>ROBUST SCALER :<b>
Here , just like standard scaler the statistical values are stored and the next samples
are scaled on that basis. The difference here is the median and interquartile ranges
are stored in this case. The scaling is done between 25th quantile and 75th quantile.
<li><b>QUARTILE TRANSFORMER :<b>
This type of scaling method follows a uniform or normal distribution. So when this
is applied on a feature ,this tends to spread out the most frequently occurring
observation. So , it is suppose to reduce the outliers impact on the data.
<li><b>NORMALIZER :<b>
This also scales the data between [0,1].
</li></ol>
<li><b>Other Methods :<b>
<ol>
<li><b>STANDARD DEVIATION :<b>
In this method , we calculate the mean and standard deviation of the data frame
and subtracted the mean from the data points and divided that by standard
deviation.
<li><b>QUARTILE METHOD :<b>
In this we calculate , the 25th quantile and 75th quantile with which we can
calculate the IQR which is the difference between these two.<br>
min = q25 - (iqr*1.5) = -25<br>
max = q75 + (iqr*1.5) = 175<br>
So anything between min and max will be considered as outliers. As most of our
data is greater than 175 , there will be lot of anomalies. So , we are eliminating it.
<li><b>ELIMINATING /DELETING OUTLIERS :<b>
This is similar to standard deviation method , but in this what we do is that we
change the number by which we want to subtract mean from the value.
</li></ol>

      
    
