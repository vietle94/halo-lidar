DOCUMENTATION FOR THIS REPOSITORY

halo_data.py contains all functions and classes used to analyze data. Import it
to use in your code.

```python
import halo_data.py
```
There are two stand-alone functions:

- Get a list of .nc files from a path

```python
hd.getdata()
```

- Get (indice -1) of the specified date from that path

```python
hd.getdata_date(data_list, pick_date = '20161230')
```

- Load data as halo_data class with

```python
hd.halo_data(data_path)
```
Some useful attributes and methods

```python
df = hd.halo_data(data_path)

df.info
df.full_data
df.full_data_names
df.data
df.data_names
df.more_info

# Get meta data of each variable
df.meta_data('co_signal')

# Get meta data of all variables
{'==>' + key: df.meta_data(key) for key in df.full_data_names}

# Only crucial info
{'==>' + key: df.meta_data(key)['_attributes'] for key in df.full_data_names}
```
