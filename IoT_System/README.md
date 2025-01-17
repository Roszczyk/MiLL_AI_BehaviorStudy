### Secret passwords and tokens

To use passwords and tokens create a python file named passwords_gitignore.py. The contents of this file should look like this:

```sh  
def get_token():
    return "<your token>"

def get_org():
    return "<your org>"

def get_mqtt_password():
    return "<your mqtt password>"
  ```