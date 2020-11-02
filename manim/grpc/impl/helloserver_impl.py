from ..gen import helloserver_pb2
from ..gen import helloserver_pb2_grpc


class Greeter(helloserver_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        return helloserver_pb2.HelloReply(message="Hello, %s!" % request.name)

    def SayHelloAgain(self, request, context):
        return helloserver_pb2.HelloReply(message="Hello again, %s!" % request.name)
