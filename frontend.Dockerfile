FROM node:21-alpine as build

WORKDIR /app

COPY frontend/prototype/package.json frontend/prototype/package-lock.json ./

RUN npm install

COPY frontend/prototype/ ./

RUN npm run build

EXPOSE 80
EXPOSE 3000
