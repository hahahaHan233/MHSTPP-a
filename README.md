# Hawkess-POI

## 1.Dataset preprocess

- POI and user share the same index space to prevent index collision in the embedding layer.
    
    POI index: [0,POI_num-1];

    User index: [POI_num,POI_num + user_num-1].

[preprocess.py](./preprocess.py)
```python
    df['uid'] = df['uid'] + len(df['poi'].unique())
    assert len(df['uid'].unique()) + len(df['poi'].unique()) == df['uid'].max() + 1
```


## 2.Model

- Haversine Distance

https://www.movable-type.co.uk/scripts/latlong.html
https://www.vcalc.com/wiki/vcalc/haversine-distance


- Training

- Evaluate
  