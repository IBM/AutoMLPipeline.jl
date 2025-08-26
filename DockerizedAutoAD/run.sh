docker build -t automlad --platform=linux/amd64 .
docker run -it --rm --platform=linux/amd64 automlad

# julia --project -- ./main.jl -c high -t regression -f 3 -w 7 iris_reg.csv
# julia --project -- ./main.jl -c low -t classification -f 3 -w 3 iris.csv
# julia --project -- ./main.jl -c low -t anomalydetection iris.csv
# podman run -it --rm --platform=linux/amd64 localhost/automlai -u http://spendor2.sl.cloud9.ibm.com:30412 iris.csv
podman run -it --rm --platform=linux/amd64 localhost/automlad -u http://spendor3.sl.cloud9.ibm.com:30412 ../AutoAD/data/node_cpu_ratio_rate_5m_1d_1m.csv
