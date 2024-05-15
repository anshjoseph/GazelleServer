commit="new added"
export commit

pushGPUv:
	git add .
	git commit -m $(commit)
	git push -u origin GPUversion 
pullGPUv:
	git pull origin GPUversion
simpleClientStart:
	clear
	python3 client.py
smlClientStart:
	clear
	python3 client01.py