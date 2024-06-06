# Installation

```shell
$ git clone https://github.com/barhanc/softeng-proj.git
$ cd softeng-proj
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
(venv) $ python3 app.py
```

To install as a one-file executable: in file `app.py` change `reload=True` to `reload=False` in
`ui.run(...)` and depending on whether you want a native or browse app set `native=True` or
`native=False`. See: [here](https://nicegui.io/documentation/section_configuration_deployment).