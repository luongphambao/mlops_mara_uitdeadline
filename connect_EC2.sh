chmod 400 private_key.pem
ssh -i private_key.pem ubuntu@13.229.118.139


ssh -L 5000:localhost:5000 -i private_key.pem ubuntu@13.229.118.139 #connect to EC2 and forward port 5000 to local host 5000
