FROM node:14-alpine as build

WORKDIR /app

COPY Frontend/package.json Frontend/package-lock.json ./

RUN npm install

COPY Frontend ./

RUN npm run build

FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 80
EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
