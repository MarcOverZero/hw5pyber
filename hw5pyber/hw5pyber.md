

```python
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline

```


```python
city = pd.read_csv('raw_data/city_data.csv')
ride = pd.read_csv('raw_data/ride_data.csv')
```


```python
##  Average Fare ($) Per City
average_fare = ride.groupby(['city'])['fare'].mean().map('${:.2f}'.format).sort_values()
average_fare.reset_index()[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>South Latoya</td>
      <td>$20.09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>West Gabriel</td>
      <td>$20.35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Royland</td>
      <td>$20.57</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Leahton</td>
      <td>$21.24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Raymondhaven</td>
      <td>$21.48</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Total Number of Rides Per City
ride_count = ride.groupby('city')['fare'].count().sort_values(ascending = False)
ride_count.reset_index()[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>West Angela</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>South Karenland</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>North Jason</td>
      <td>35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Port Frank</td>
      <td>33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Liumouth</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Total Number of Drivers Per City
city.groupby('city')['driver_count']
city.drop('type', 1)[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>driver_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Richardfort</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Williamsstad</td>
      <td>59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Port Angela</td>
      <td>67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rodneyfort</td>
      <td>34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>West Robert</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>




```python
city_ride_join = pd.merge(city, ride, how='inner', on = 'city')
city_ride_join.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>driver_count</th>
      <th>type</th>
      <th>date</th>
      <th>fare</th>
      <th>ride_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Richardfort</td>
      <td>38</td>
      <td>Urban</td>
      <td>2018-02-24 08:40:38</td>
      <td>13.93</td>
      <td>5628545007794</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Richardfort</td>
      <td>38</td>
      <td>Urban</td>
      <td>2018-02-13 12:46:07</td>
      <td>14.00</td>
      <td>910050116494</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Richardfort</td>
      <td>38</td>
      <td>Urban</td>
      <td>2018-02-16 13:52:19</td>
      <td>17.92</td>
      <td>820639054416</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Richardfort</td>
      <td>38</td>
      <td>Urban</td>
      <td>2018-02-01 20:18:28</td>
      <td>10.26</td>
      <td>9554935945413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Richardfort</td>
      <td>38</td>
      <td>Urban</td>
      <td>2018-04-17 02:26:37</td>
      <td>23.00</td>
      <td>720020655850</td>
    </tr>
  </tbody>
</table>
</div>




```python
# grouped cities
cities = city_ride_join.groupby('city')
# average fare by city
average_fare = cities['fare'].mean()
# rides per city
ride_count = cities['type'].count()
# sum of drivers per city
drivers = cities['driver_count'].max()
# city type (Urban, Suburban, Rural)
city_type = cities['type'].max()

```


```python
#df merge
ride_invs = pd.DataFrame({'average fare': average_fare, '# of rides': ride_count,'drivers': drivers,'city type': city_type})
ride_invs.reset_index().nlargest(10, "drivers")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>average fare</th>
      <th># of rides</th>
      <th>drivers</th>
      <th>city type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>116</th>
      <td>West Samuelburgh</td>
      <td>21.767600</td>
      <td>25</td>
      <td>73</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>96</th>
      <td>South Michelleport</td>
      <td>24.451613</td>
      <td>31</td>
      <td>72</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>105</th>
      <td>West Anthony</td>
      <td>24.736667</td>
      <td>30</td>
      <td>70</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Liumouth</td>
      <td>26.150000</td>
      <td>33</td>
      <td>69</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Port Angela</td>
      <td>23.836842</td>
      <td>19</td>
      <td>67</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Reynoldsfurt</td>
      <td>21.919474</td>
      <td>19</td>
      <td>67</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>13</th>
      <td>East Kaylahaven</td>
      <td>23.757931</td>
      <td>29</td>
      <td>65</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Jerryton</td>
      <td>25.649200</td>
      <td>25</td>
      <td>64</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Royland</td>
      <td>20.570667</td>
      <td>30</td>
      <td>64</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Grahamburgh</td>
      <td>25.221200</td>
      <td>25</td>
      <td>61</td>
      <td>Urban</td>
    </tr>
  </tbody>
</table>
</div>




```python
ride_invs.groupby('city type').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>average fare</th>
      <th># of rides</th>
      <th>drivers</th>
    </tr>
    <tr>
      <th>city type</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rural</th>
      <td>34.637765</td>
      <td>6.944444</td>
      <td>4.333333</td>
    </tr>
    <tr>
      <th>Suburban</th>
      <td>30.737298</td>
      <td>17.361111</td>
      <td>13.611111</td>
    </tr>
    <tr>
      <th>Urban</th>
      <td>24.499122</td>
      <td>24.621212</td>
      <td>36.439394</td>
    </tr>
  </tbody>
</table>
</div>




```python
ride_invs.groupby("city type").count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>average fare</th>
      <th># of rides</th>
      <th>drivers</th>
    </tr>
    <tr>
      <th>city type</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rural</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>Suburban</th>
      <td>36</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>Urban</th>
      <td>66</td>
      <td>66</td>
      <td>66</td>
    </tr>
  </tbody>
</table>
</div>




```python
## bubble plot components
city_sort = ride_invs.groupby("city type")
urban = city_sort.get_group('Urban')
urban_total_rides = urban['# of rides']
urban_average_fare = urban['average fare']
urban_drivers = urban['drivers']

suburban = city_sort.get_group('Suburban')
suburban_total_rides = suburban['# of rides']
suburban_average_fare = suburban['average fare']
suburban_drivers = suburban['drivers']

rural = city_sort.get_group('Rural')
rural_total_rides = rural['# of rides']
rural_average_fare = rural['average fare']
rural_drivers = rural['drivers']

```


```python
sb.set()
plt.figure(figsize=(10,8))
rural_plot = plt.scatter(rural_total_rides, rural_average_fare, s = rural_drivers*10, 
            c = 'gold', alpha = 0.5, edgecolor='black',linewidths=.5)
urban_plot = plt.scatter(urban_total_rides, urban_average_fare, s = urban_drivers*10, 
            c = 'lightcoral', alpha = 0.5, edgecolor='black',linewidths=.5)
suburban_plot = plt.scatter(suburban_total_rides, suburban_average_fare, s = suburban_drivers*10, 
            c = 'lightskyblue', alpha = 0.5, edgecolor='black',linewidths=.5)


plt.title('2016 Pyber Ride Share Data', fontsize = 14)
plt.xlabel('Number of Rides Per City', fontsize = 12)
plt.ylabel('Average Fare USD', fontsize = 12)
lgnd = plt.legend(('rural','urban','suburban'), loc ='best',title = 'city types')

plt.show()
```


![png](output_11_0.png)



```python
# % of Total Fares by City Type
fare = city_ride_join.groupby('type')['fare'].sum()
fare_df = pd.DataFrame({'Fare by city': fare})
total_fare = city_ride_join['fare'].sum()
fare_df['percentage'] = ((fare_df/total_fare)*100)
fare_df['Fare by city'] = fare_df['Fare by city']
fare_df.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>Fare by city</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rural</td>
      <td>4327.93</td>
      <td>6.811493</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Suburban</td>
      <td>19356.33</td>
      <td>30.463872</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Urban</td>
      <td>39854.38</td>
      <td>62.724635</td>
    </tr>
  </tbody>
</table>
</div>




```python
### pie charts

#pie formatting
colors = 'gold', 'lightskyblue', 'lightcoral'
labels = 'Rural', 'Suburbabn', 'Urban'
explode = (0, 0, .11)

# % of Total Fares by City Type
rural_fare_percent = fare_df.iloc[0]['percentage']
suburban_fare_percent = fare_df.iloc[1]['percentage']
urban_fare_percent = fare_df.iloc[2]['percentage']

fares = [rural_fare_percent, suburban_fare_percent, urban_fare_percent]

plt.pie(fares, labels=labels, colors=colors, explode=explode, startangle=450, shadow=True, autopct="%1.1f%%")
plt.title("% of Total Fares by City Type", fontsize=14)
plt.show()
```


![png](output_13_0.png)



```python
# of Total Rides by City Type
ride = city_ride_join.groupby('type')['ride_id'].count()
ride_df = pd.DataFrame({'Total Rides by city': ride})
total_rides = city_ride_join['ride_id'].count()
ride_df['percentage'] = ((ride_df/total_rides)*100)
ride_df.reset_index()

#fare precent variables
rural_ride_percent = ride_df.iloc[0]['percentage']
suburban_ride_percent = ride_df.iloc[1]['percentage']
urban_ride_percent = ride_df.iloc[2]['percentage']

rides = [rural_ride_percent, suburban_ride_percent, urban_ride_percent]

plt.pie(rides, labels=labels, colors=colors, explode=explode, startangle=90, shadow=True, autopct="%1.1f%%")
plt.title("% of Total Rides by City Type", fontsize=14)
plt.show()
```


![png](output_14_0.png)



```python
# % of Total Drivers by City Type
drivers = ride_invs.groupby("city type")['drivers'].sum()
drivers_df = pd.DataFrame({'Total Drivers by city': drivers})
total_drivers = ride_invs['drivers'].sum()
drivers_df['percentage'] = ((drivers_df/total_drivers)*100)
drivers_df.reset_index()

#fare precent variables
rural_driver_percent = drivers_df.iloc[0]['percentage']
suburban_driver_percent = drivers_df.iloc[1]['percentage']
urban_driver_percent = drivers_df.iloc[2]['percentage']

drivers = [rural_driver_percent, suburban_driver_percent, urban_driver_percent]

plt.pie(drivers, labels=labels, colors=colors, explode=explode, startangle=110, shadow=True, autopct="%1.1f%%")
plt.title("% of Total Drivers by City Type", fontsize=14)
plt.show()
```


![png](output_15_0.png)



```python
## observations
    # 1. Pyber is most common in Urban areas as there are the most number of drivers and most number of requested rides there.
    # 2. Rural areas averaged higher fares, longer travel distances seem a relevant component of that.
    # 3. Number of rides and fares have an inverse relationship.
```
