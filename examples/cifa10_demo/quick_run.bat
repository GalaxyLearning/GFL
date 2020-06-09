python fl_server.py

sleep 2

for ((i=0;i<2;i++))
do
    python fl_client $i
done