docker build -t automlts:v1.0 --platform=linux/amd64 .
docker run -it --rm --platform=linux/amd64 automlai

# podman run -it --rm --platform=linux/amd64 localhost/automlai -u http://spendor2.sl.cloud9.ibm.com:30412 iris.csv
# podman run -it --rm --platform=linux/amd64 localhost/automlad -u http://spendor3.sl.cloud9.ibm.com:30412 ../AutoAD/data/node_cpu_ratio_rate_5m_1d_1m.csv
# argo -n argo submit --from clusterworkflowtemplate/automlad-template -p votepercent=0.0 -p input=node_cpu_ratio_rate_5m_1d_1m.csv -p predictiontype=anomalydetection --watch --log
# docker run -it --rm --platform=linux/amd64 -v ${HOME}/phome/julia/AutoMLPipeline.jl/AutoAD/data/:/data/ ppalmes/automlad:v2.0 -v 0.0 -u http://mlflow.isiath.duckdns.org:8082 /data/node_cpu_ratio_rate_5m_1d_1m.csv

julia --project ./main.jl -p -f 20 -r 9929adf41952406188c500b19e1e73ab ./../AutoAD/data/node_cpu_ratio_rate_5m_1d_1m.csv
julia -m AutoAD -f 20 -u http://localhost:8081 ./../AutoAD/data/node_cpu_ratio_rate_5m_1d_1m.csv
