echo "parse json"
python model/dataset.py config.conf parse

echo "map id for data"
python model/dataset.py config.conf map_index

echo 'map relation for data'
python model/dataset.py config.conf map_relation

