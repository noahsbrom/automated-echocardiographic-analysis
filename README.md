# Automated Echocardiographic Analysis Tool
Some information about the app

## Starting the app
1. Ensure that python is installed
2. Run `./deploy.sh` from the app directory. 
3. The app should now be running on localhost:8080. 

## Understanding the app
* The deploy script configures a virtual environment in which
the user can run the app without installing various dependecies on their
machine.
* Image uploads are stored in static/image-uploads, a directory that is created 
on deployment and deleted on termination. By default, Flask serves static files
(like CSS stylesheets and image uploads) from a static view that takes a path relative
to the app/static directory. 
