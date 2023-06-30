import gdown 

id="1rtGvrwKCNtqTvCituUAHjY8mldMhTSbM"
output = "data_phase-1.zip"
gdown.download(id=id, output=output, quiet=False)


ssh -L 5000:localhost:5000 -i "