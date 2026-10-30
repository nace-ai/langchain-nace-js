/** The package version; a unit test keeps it equal to package.json. */
export const VERSION = "0.1.1";

/**
 * Sent as `User-Agent` and `X-Nace-Client` on every request, so Nace can count
 * traffic per integration. `-js` tells it apart from the Python package.
 */
export const CLIENT_NAME = `langchain-nace-js/${VERSION}`;
