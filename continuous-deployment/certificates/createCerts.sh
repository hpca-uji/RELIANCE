CA_PATH="./ca"
SERVER_PATH="./server"
CLIENT_PATH="./client"

rm ${CA_PATH}/*.pem
rm ${SERVER_PATH}/*.pem
rm ${CLIENT_PATH}/*.pem

# 1. Generate CA's private key and self-signed certificate
openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout ${CA_PATH}/ca-key.pem -out ${CA_PATH}/ca-cert.pem -subj "/CN=ssl-ca-cert"
#openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout ${CA_PATH}/ca-key.pem -out ${CA_PATH}/ca-cert.pem -subj "/CN=localhost"

echo "CA's self-signed certificate"
openssl x509 -in ${CA_PATH}/ca-cert.pem -noout -text

# 2. Generate web server's private key and certificate signing request (CSR)
#openssl req -newkey rsa:4096 -nodes -keyout ${SERVER_PATH}/server-key.pem -out ${SERVER_PATH}/server-req.pem -subj "/CN=ssl-ca-server-cert"
openssl req -newkey rsa:4096 -nodes -keyout ${SERVER_PATH}/server-key.pem -out ${SERVER_PATH}/server-req.pem -subj "/CN=localhost"

# 3. Use CA's private key to sign web server's CSR and get back the signed certificate
openssl x509 -req -in ${SERVER_PATH}/server-req.pem -days 365 -CA ${CA_PATH}/ca-cert.pem -CAkey ${CA_PATH}/ca-key.pem -CAcreateserial -out ${SERVER_PATH}/server-cert.pem

echo "Server's signed certificate"
openssl x509 -in ${SERVER_PATH}/server-cert.pem -noout -text

# 4. Generate client's private key and certificate signing request (CSR)
openssl req -newkey rsa:4096 -nodes -keyout ${CLIENT_PATH}/client-key.pem -out ${CLIENT_PATH}/client-req.pem -subj "/CN=ssl-ca-client-cert"

# 5. Use CA's private key to sign client's CSR and get back the signed certificate
openssl x509 -req -in ${CLIENT_PATH}/client-req.pem -days 365 -CA ${CA_PATH}/ca-cert.pem -CAkey ${CA_PATH}/ca-key.pem -CAcreateserial -out ${CLIENT_PATH}/client-cert.pem

echo "Client's signed certificate"
openssl x509 -in ${CLIENT_PATH}/client-cert.pem -noout -text

# 6. Create client's certificate chain
openssl x509 -req -days 365 -in ${CLIENT_PATH}/client-req.pem -CA ${CA_PATH}/ca-cert.pem -CAkey ${CA_PATH}/ca-key.pem -set_serial 01 -out ${CLIENT_PATH}/client-chain.pem

#7. Verify client's certificate chain with CA's public key
openssl verify -CAfile ${CA_PATH}/ca-cert.pem ${CLIENT_PATH}/client-chain.pem

# Some useful verification commands
#openssl s_client -quiet -connect localhost:8001
#openssl s_client -showcerts -connect localhost:8001

