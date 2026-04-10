#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Generate all TLS certificates, MQTT auth files, Django secrets, and DB
# passwords required by the SceneScape + Loss Prevention stack.

set -e

CERTDOMAIN="scenescape.intel.com"
CERTPASS=$(openssl rand -base64 33)
DBPASS=${DBPASS:-"$(openssl rand -base64 12)"}
EXEC_PATH="$(dirname "$(readlink -f "$0")")"
MQTTUSERS="controller.auth=scenectrl browser.auth=webuser calibration.auth=calibuser"
SECRETSDIR="$EXEC_PATH"

echo "=== Generating SceneScape secrets ==="

# ---- Root CA ----
echo "Generating root CA key..."
mkdir -p "$SECRETSDIR/ca"
openssl ecparam -name secp384r1 -genkey | openssl ec -aes256 -passout pass:"$CERTPASS" \
    -out "$SECRETSDIR/ca/scenescape-ca.key"

echo "Generating root CA certificate..."
mkdir -p "$SECRETSDIR/certs"
openssl req -passin pass:"$CERTPASS" -x509 -new -key "$SECRETSDIR/ca/scenescape-ca.key" -days 1825 \
    -out "$SECRETSDIR/certs/scenescape-ca.pem" -subj "/CN=ca.$CERTDOMAIN"

# ---- Web key + certificate ----
echo "Generating web key..."
openssl ecparam -name secp384r1 -genkey -noout -out "$SECRETSDIR/certs/scenescape-web.key"
echo "Generating CSR for web.$CERTDOMAIN..."
openssl req -new -out "$SECRETSDIR/certs/scenescape-web.csr" -key "$SECRETSDIR/certs/scenescape-web.key" \
    -config <(sed -e "s/##CN##/web.$CERTDOMAIN/" -e "s/##SAN##/DNS.1=web.$CERTDOMAIN/" \
    -e "s/##KEYUSAGE##/serverAuth/" "$EXEC_PATH/openssl.cnf")
echo "Generating certificate for web.$CERTDOMAIN..."
openssl x509 -passin pass:"$CERTPASS" -req -in "$SECRETSDIR/certs/scenescape-web.csr" \
    -CA "$SECRETSDIR/certs/scenescape-ca.pem" -CAkey "$SECRETSDIR/ca/scenescape-ca.key" -CAcreateserial \
    -out "$SECRETSDIR/certs/scenescape-web.crt" -days 360 -extensions x509_ext -extfile \
    <(sed -e "s/##SAN##/DNS.1=web.$CERTDOMAIN/" -e "s/##KEYUSAGE##/serverAuth/" "$EXEC_PATH/openssl.cnf")

# ---- Broker key + certificate ----
echo "Generating broker key..."
openssl ecparam -name secp384r1 -genkey -noout -out "$SECRETSDIR/certs/scenescape-broker.key"
echo "Generating CSR for broker.$CERTDOMAIN..."
openssl req -new -out "$SECRETSDIR/certs/scenescape-broker.csr" -key "$SECRETSDIR/certs/scenescape-broker.key" \
    -config <(sed -e "s/##CN##/broker.$CERTDOMAIN/" -e "s/##SAN##/DNS.1=broker.$CERTDOMAIN/" \
    -e "s/##KEYUSAGE##/serverAuth/" "$EXEC_PATH/openssl.cnf")
echo "Generating certificate for broker.$CERTDOMAIN..."
openssl x509 -passin pass:"$CERTPASS" -req -in "$SECRETSDIR/certs/scenescape-broker.csr" \
    -CA "$SECRETSDIR/certs/scenescape-ca.pem" -CAkey "$SECRETSDIR/ca/scenescape-ca.key" -CAcreateserial \
    -out "$SECRETSDIR/certs/scenescape-broker.crt" -days 360 -extensions x509_ext -extfile \
    <(sed -e "s/##SAN##/DNS.1=broker.$CERTDOMAIN/" -e "s/##KEYUSAGE##/serverAuth/" "$EXEC_PATH/openssl.cnf")

# ---- Autocalibration key + certificate ----
echo "Generating autocalibration key..."
openssl ecparam -name secp384r1 -genkey -noout -out "$SECRETSDIR/certs/scenescape-autocalibration.key"
echo "Generating CSR for autocalibration.$CERTDOMAIN..."
openssl req -new -out "$SECRETSDIR/certs/scenescape-autocalibration.csr" -key "$SECRETSDIR/certs/scenescape-autocalibration.key" \
    -config <(sed -e "s/##CN##/autocalibration.$CERTDOMAIN/" -e "s/##SAN##/DNS.1=autocalibration.$CERTDOMAIN/" \
    -e "s/##KEYUSAGE##/serverAuth/" "$EXEC_PATH/openssl.cnf")
echo "Generating certificate for autocalibration.$CERTDOMAIN..."
openssl x509 -passin pass:"$CERTPASS" -req -in "$SECRETSDIR/certs/scenescape-autocalibration.csr" \
    -CA "$SECRETSDIR/certs/scenescape-ca.pem" -CAkey "$SECRETSDIR/ca/scenescape-ca.key" -CAcreateserial \
    -out "$SECRETSDIR/certs/scenescape-autocalibration.crt" -days 360 -extensions x509_ext -extfile \
    <(sed -e "s/##SAN##/DNS.1=autocalibration.$CERTDOMAIN/" -e "s/##KEYUSAGE##/serverAuth/" "$EXEC_PATH/openssl.cnf")

# ---- VDMS client key + certificate ----
echo "Generating VDMS client key..."
openssl ecparam -name secp384r1 -genkey -noout -out "$SECRETSDIR/certs/scenescape-vdms-c.key"
echo "Generating CSR for vdms-client.$CERTDOMAIN..."
openssl req -new -out "$SECRETSDIR/certs/scenescape-vdms-c.csr" -key "$SECRETSDIR/certs/scenescape-vdms-c.key" \
    -config <(sed -e "s/##CN##/vdms-client.$CERTDOMAIN/" -e "s/##SAN##/DNS.1=vdms-client.$CERTDOMAIN/" \
    -e "s/##KEYUSAGE##/clientAuth/" "$EXEC_PATH/openssl.cnf")
echo "Generating certificate for vdms-client.$CERTDOMAIN..."
openssl x509 -passin pass:"$CERTPASS" -req -in "$SECRETSDIR/certs/scenescape-vdms-c.csr" \
    -CA "$SECRETSDIR/certs/scenescape-ca.pem" -CAkey "$SECRETSDIR/ca/scenescape-ca.key" -CAcreateserial \
    -out "$SECRETSDIR/certs/scenescape-vdms-c.crt" -days 360 -extensions x509_ext -extfile \
    <(sed -e "s/##SAN##/DNS.1=vdms-client.$CERTDOMAIN/" -e "s/##KEYUSAGE##/clientAuth/" "$EXEC_PATH/openssl.cnf")

# ---- Django secrets ----
echo "Generating Django secrets..."
mkdir -p "$SECRETSDIR/django"
echo -n SECRET_KEY= > "$SECRETSDIR/django/secrets.py"
python3 -c 'import secrets; print("\x27" + "".join([secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)") \
    for i in range(50)]) + "\x27")' >> "$SECRETSDIR/django/secrets.py"
echo "DATABASE_PASSWORD='$DBPASS'" >> "$SECRETSDIR/django/secrets.py"

# ---- PostgreSQL env ----
mkdir -p "$SECRETSDIR/pgserver"
echo "POSTGRES_PASSWORD=\"$DBPASS\"" > "$SECRETSDIR/pgserver/pgserver.env"

# ---- MQTT auth files ----
echo "Generating MQTT auth files..."
for uid in $MQTTUSERS; do
    JSONFILE=${uid%=*}
    USERPASS=${uid##*=}
    case $USERPASS in
        *:* ) ;;
        * ) USERPASS=$USERPASS:$(openssl rand -base64 12);;
    esac
    USER=${USERPASS%:*}
    PASS=${USERPASS##*:}
    echo '{"user": "'$USER'", "password": "'$PASS'"}' > "$SECRETSDIR/$JSONFILE"
done

# ---- SUPASS (admin password) ----
echo "Generating SUPASS..."
SUPASS=$(openssl rand -base64 16)
echo -n "$SUPASS" > "$SECRETSDIR/supass"

echo "=== Secrets generated in $SECRETSDIR ==="
