FROM node:lts-alpine as builder

WORKDIR /app

COPY frontend/prototype/package.json frontend/prototype/package-lock.json ./

RUN npm install

COPY frontend/prototype/ ./

RUN npm run build

FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80
EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]