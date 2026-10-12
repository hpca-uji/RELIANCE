FROM node:18.16.0

# Create app directory
WORKDIR /usr/src/app

# Update and install libwrap.so.0
RUN apt-get update
RUN apt-get install libwrap0

# Update and set timezone to Madrid
RUN apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Set max space size in memory
#ENV NODE_OPTIONS=--max_old_space_size=10240

RUN apt-get install nano

# Copy app dependencies
COPY package*.json ./
# Update app dependencies
#RUN npm install -g npm-check-updates
#RUN ncu --upgrade
# Install app dependencies
RUN npm install

# If you are building your code for production
# RUN npm ci --only=production

# Bundle app source
#COPY . .
COPY LICENSE ./
COPY README.md ./
COPY ./bin ./bin
COPY ./config ./config
COPY ./src ./src
COPY ./public ./public

# Set the user to use when running this image
#USER node

# Create needed directories
RUN mkdir ./logs
RUN mkdir ./data
RUN mkdir ./profiler
RUN mkdir ./src/certs/
RUN mkdir ./src/certs/sessions

CMD [ "/bin/bash", "-c", "npm run https-certs;npm run session-certs;npm start"]
