FROM node:14-alpine as build

WORKDIR /app

COPY frontend/prototype/package.json frontend/prototype/package-lock.json ./

RUN npm install

COPY frontend/prototype/ ./

RUN npm run build

FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
