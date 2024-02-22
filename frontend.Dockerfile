FROM node:alpine

WORKDIR /app

COPY frontend/prototype/package*.json ./

RUN npm install

COPY frontend/prototype/ .

EXPOSE 3000

CMD ["npm", "start"]