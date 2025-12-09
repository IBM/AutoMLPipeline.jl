kubectl -n prometheus port-forward svc/thanos-query 9090

curl 'http://localhost:9090/api/v1/query?query={cluster:node_cpu:ratio_rate5m}\[2d:1m\]' |
	jq '.data.result[].values[][1]' |
	sed 's/\"//g' >node_cpu_ratio_rate_5m_2d_1m.csv

curl 'http://localhost:9090/api/v1/query?query={cluster:node_cpu:ratio_rate5m}\[1d:1m\]' |
	jq '.data.result[].values[][1]' |
	sed 's/\"//g' >node_cpu_ratio_rate_5m_1d_1m.csv

curl 'http://localhost:9090/api/v1/query?query=node_memory_Active_bytes\[2d:1m\]' >node_memory_Active_bytes_2d_1m.csv
