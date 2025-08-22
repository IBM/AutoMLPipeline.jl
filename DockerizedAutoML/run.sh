docker build -t automlai --platform=linux/amd64 .
docker run -it --rm --platform=linux/amd64 automlai

# julia --project -- ./main.jl -c high -t regression -f 3 -w 7 iris_reg.csv
# julia --project -- ./main.jl -c low -t classification -f 3 -w 3 iris.csv
# julia --project -- ./main.jl -c low -t anomalydetection iris.csv
# podman run -it --rm --platform=linux/amd64 localhost/automlai -u http://spendor2.sl.cloud9.ibm.com:30412 iris.csv
# julia --project -- ./main.jl -c low -t classification -f 3 -w 3 iris.csv --predict_only --runid cd4e463d6a414aa4aaad173e567d7d22 -o /tmp/hello.txt
