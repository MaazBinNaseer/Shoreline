// middleware.js
export function middleware(request) {
    const auth = request.headers.get("authorization");
  
    const username = "admin";
    const password = "secret123";
    const basicAuth = btoa(`${username}:${password}`);
  
    if (auth === `Basic ${basicAuth}`) {
      return new Response(null, { status: 200 });
    }
  
    return new Response("Unauthorized", {
      status: 401,
      headers: {
        "WWW-Authenticate": 'Basic realm="Protected Area"',
      },
    });
  }
  