docker build -t automlai:v3.0 --platform=linux/amd64 .
docker run -it --rm --platform=linux/amd64 automlai:v3.0

# julia --project -- ./main.jl -c high -t regression -f 3 -w 7 iris_reg.csv
# julia --project -- ./main.jl -c low -t classification -f 3 -w 3 iris.csv
# julia --project -- ./main.jl -c low -t anomalydetection iris.csv
# podman run -it --rm --platform=linux/amd64 localhost/automlai -u http://spendor2.sl.cloud9.ibm.com:30412 iris.csv
# podman run -it --rm -v `pwd`:/data/  localhost/automlai -u http://spendor2.sl.cloud9.ibm.com:30412 -t regression /data/iris_reg.csv
# julia --project -- ./main.jl -c low -t classification -f 3 -w 3 iris.csv --predict_only --runid cd4e463d6a414aa4aaad173e567d7d22 -o /tmp/hello.txt

julia --project -- ./main.jl -t regression -u http://mlflow.isiath.duckdns.org:8082 -p -r 064fb7a188d34a3da87f2271b8d8d9c2 -o /tmp/reg.txt ./iris_reg.csv
julia --project -- ./main.jl -t classification -u http://mlflow.isiath.duckdns.org:8082 -p -r 8dbea59123ec469db3ee7b807b3ab6d9 -o /tmp/class.txt ./iris.csv

docker run -it --rm -v $(pwd):/data/ localhost/automlai:v3.0 -u http://mlflow.isiath.duckdns.org:8082 -t regression /data/iris_reg.csv
docker run -it --rm -v $(pwd):/data/ localhost/automlai:v3.0 -u http://mlflow.isiath.duckdns.org:8082 -t classification /data/iris.csv

docker run -it --rm -v $(pwd):/data/ localhost/automlai:v3.0 -u http://mlflow.isiath.duckdns.org:8082 -t regression -p -r 064fb7a188d34a3da87f2271b8d8d9c2 /data/iris_reg.csv
docker run -it --rm -v $(pwd):/data/ localhost/automlai:v3.0 -u http://mlflow.isiath.duckdns.org:8082 -t classification -p -r 8dbea59123ec469db3ee7b807b3ab6d9 /data/iris.csv

# notes to use CondaPkg to update/install python modules during docker build
julia -m AutoAI -u http://mlflow.isiath.duckdns.org:8082 -t regression ../DockerizedAutoML/iris_reg.csv
julia -m AutoAI -u http://mlflow.isiath.duckdns.org:8082 -t classification ~/tmp/occupancy-detection/occupancy_training.csv
