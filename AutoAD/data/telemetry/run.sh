curl 'http://localhost:9090/api/v1/query?query={cluster:node_cpu:ratio_rate5m}\[1d:1m\]' | jq '.data.result[].values[][1]' | sed 's/\"//g' >node_cpu_ratio_rate_5m_1d_1m.csv
