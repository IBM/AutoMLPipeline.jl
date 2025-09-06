argo -n argo submit --from clusterworkflowtemplate/automlai-dualsearch \
	-p workers=20 -p input=iris.csv \
	-p predictiontype=classification --watch --log

argo -n argo submit --from clusterworkflowtemplate/automlai-dualsearch \
	-p workers=20 -p input=iris_reg.csv \
	-p predictiontype=regression --watch --log

argo -n argo submit --from clusterworkflowtemplate/automlad-template \
	-p votepercent=0.0 -p input=node_cpu_ratio_rate_5m_1d_1m.csv -p predictiontype=anomalydetection --watch --log
