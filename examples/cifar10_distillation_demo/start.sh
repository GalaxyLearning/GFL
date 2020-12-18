for client_id in {1..10}
do
    python fl_client_$client_id.py
    printf "execute fl_client_$client_id.py success"
done