chmod +x run_attack.sh
zip $1 ./nets/* ./preprocessing/* ./*.py metadata.json run_attack.sh ./model_ckpts/*

