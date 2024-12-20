import fs from 'fs';
import path from 'path';
import PDFParser from 'pdf2json';
import fetch from 'node-fetch'; // Ensure you have node-fetch installed
import mammoth from 'mammoth'; // Assuming you have mammoth for DOCX parsing

// Function to extract text from the PDF
const extractTextFromPdf = (pdfBuffer) => {
    return new Promise((resolve, reject) => {
        const pdfParser = new PDFParser();

        // Handle errors
        pdfParser.on("pdfParser_dataError", (errData) => {
            reject(`Error parsing PDF: ${errData.parserError}`);
        });

        // When parsing is done, extract the text content
        pdfParser.on("pdfParser_dataReady", (pdfData) => {
            // pdfParser.getRawTextContent() gives us the plain text of the PDF
            const textContent = pdfParser.getRawTextContent();
            resolve(textContent);  // Resolve with extracted text
        });

        // Load the PDF from buffer (not file path)
        pdfParser.parseBuffer(pdfBuffer); // Use parseBuffer instead of loadPDF
    });
};

// Function to fetch and parse resumes
const fetchResumeText = async (resumeUrl) => {
    try {
        // Fetch the file as a buffer
        const response = await fetch(resumeUrl);
        const buffer = await response.arrayBuffer();
        const fileBuffer = Buffer.from(buffer);

        // Determine file type from URL
        const fileType = resumeUrl.split('.').pop().toLowerCase();

        let resumeText = "";

        if (fileType === "pdf") {
            // Parse PDF with custom rendering
            resumeText = await extractTextFromPdf(fileBuffer);
        } else if (fileType === "docx") {
            // Parse DOCX
            const docxData = await mammoth.extractRawText({ buffer: fileBuffer });
            resumeText = docxData.value; // Extracted text from DOCX
        } else {
            throw new Error("Unsupported file type. Only PDF and DOCX are supported.");
        }

        return resumeText;

    } catch (error) {
        console.error("Error fetching or parsing resume text:", error);
        return ""; // Return empty string if parsing fails
    }
};

export { fetchResumeText };
