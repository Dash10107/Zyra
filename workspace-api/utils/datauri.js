import DataUriParser from "datauri/parser.js"

import path from "node:path";

const getDataUri = (file) => {
    
    if (file !== null || file !== undefined) {
    const parser = new DataUriParser();
    const extName = path.extname(file?.originalname).toString();
    return parser.format(extName, file?.buffer);
    }

}

export default getDataUri;