FROM node:alpine

WORKDIR /app

COPY frontend/prototype/package*.json ./

RUN npm install

COPY frontend/prototype/ .

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]