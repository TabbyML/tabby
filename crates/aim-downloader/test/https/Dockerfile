FROM nginx:stable-alpine

RUN apk -U upgrade \
    && apk add --no-cache \
        apache2-utils \
        netcat-openbsd \
        openssl

RUN mkdir -p /srv \
    && chmod a+w /srv \
    && printf "user:$(openssl passwd -crypt pass)\n" > /etc/nginx/.htpasswd;

EXPOSE 80
HEALTHCHECK --interval=3s --retries=20 --timeout=1s \
  CMD nc -z localhost 80

COPY nginx.conf /etc/nginx/conf.d/default.conf

CMD nginx -g "daemon off;"
