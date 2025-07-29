# Use official nginx base image
FROM nginx:alpine

# Remove default nginx website
RUN rm -rf /usr/share/nginx/html/*

# Copy HTML and image to nginx public directory
COPY index.html /usr/share/nginx/html/index.html
COPY logo_small.png /usr/share/nginx/html/logo_small.png

# Expose port 80
EXPOSE 80

# Start nginx server
CMD ["nginx", "-g", "daemon off;"]