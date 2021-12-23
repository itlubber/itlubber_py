import os
import json
import tornado
from tornado.web import StaticFileHandler, url


class BaseHandler(tornado.web.RequestHandler):
    json_args = {}

    def data_received(self, chunk: bytes):
        pass

    def prepare(self):
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            self.json_args = json.loads(self.request.body)

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Content-Type", "application/json; charset=UTF-8")

    def options(self):
        self.set_status(204)
        self.finish()


class TestHandler(BaseHandler):
    
    def post(self):
        content = self.json_args.get("content")
        
        res = {"content": content, "result": "this is a test api."}

        self.write(dict(code=200, msg="success", data=res))


urls = [
    url(r"/api/test_handler", TestHandler, name="event_importance"),
    url(r"/(.*)", StaticFileHandler, dict(path=os.path.join(os.path.dirname(__file__), "static"), default_filename="index.html"))
]


app = tornado.web.Application(urls)


if __name__ == '__main__':    
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8000)
    tornado.ioloop.IOLoop.current().start()
