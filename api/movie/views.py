from rest_framework.views import APIView
from rest_framework.response import Response
from movie.recommendation import Recommendation

class MovieReccommendation(APIView):
    def get(self, request, *args, **kwargs):
        movie_name = request.GET.get('name', None)
        if movie_name is None:
            result = {
                "error": {
                    "message": "Parameter 'name' is required"
                }
            }
        else:
            movie_list = Recommendation.predict(movie_name)
            if movie_list is not None:
                result = {
                    "data": movie_list
                }
            else:
                result = {
                    "data": "Your movie did not related to any movie in our list"
                }
        return Response(result)

    def post(self, request, *args, **kwargs):
        Recommendation.training()
        result = {
            "data": "Train the model successfully"
        }
        return Response(result)