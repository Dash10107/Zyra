import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';

const protectedRoute = createRouteMatcher([
  '/startup/',
  '/startup/upcoming',
  '/startup/meeting(.*)',
  '/startup/previous',
  '/startup/recordings',
  '/startup/personal-room',
]);

export default clerkMiddleware((auth, req) => {
  // if (protectedRoute(req)) auth().protect();
});

export const config = {
  matcher: ['/((?!.+\\.[\\w]+$|_next).*)', '/', '/(api|trpc)(.*)'],
};
