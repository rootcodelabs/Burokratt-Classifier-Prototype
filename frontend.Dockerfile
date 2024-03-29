FROM node:18-alpine3.18
WORKDIR /app
COPY frontend/prototype/package*.json ./
RUN npm install
RUN npm install react-bootstrap bootstrap
COPY frontend/prototype/ .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
