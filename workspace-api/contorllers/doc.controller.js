import Document from "../models/doc.model.js";

export const createDocument = async (req, res) => {
    try {
        const { title, workspaceId, uploadedBy, fileUrl, description, tags } = req.body;

        const newDocument = new Document({
            title,
            workspaceId,
            uploadedBy,
            fileUrl,
            description,
            tags,
        });

        await newDocument.save();

        res.status(201).json({
            success: true,
            message: "Document uploaded successfully",
            document: newDocument,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to upload document",
            error: error.message,
        });
    }
};

export const getDocumentsByWorkspace = async (req, res) => {
    try {
        const { workspaceId } = req.params;

        const documents = await Document.find({ workspaceId }).sort({ createdAt: -1 });

        res.status(200).json({
            success: true,
            documents,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch documents",
            error: error.message,
        });
    }
};

export const getDocumentDetails = async (req, res) => {
    try {
        const { id } = req.params;

        const document = await Document.findById(id);

        if (!document) {
            return res.status(404).json({
                success: false,
                message: "Document not found",
            });
        }

        res.status(200).json({
            success: true,
            document,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch document details",
            error: error.message,
        });
    }
};

export const updateDocument = async (req, res) => {
    try {
        const { id } = req.params;
        const { title, fileUrl, description, tags } = req.body;

        const updatedDocument = await Document.findByIdAndUpdate(
            id,
            {
                title,
                fileUrl,
                description,
                tags,
            },
            { new: true }
        );

        if (!updatedDocument) {
            return res.status(404).json({
                success: false,
                message: "Document not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Document updated successfully",
            document: updatedDocument,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to update document",
            error: error.message,
        });
    }
};

export const deleteDocument = async (req, res) => {
    try {
        const { id } = req.params;

        const document = await Document.findByIdAndDelete(id);

        if (!document) {
            return res.status(404).json({
                success: false,
                message: "Document not found",
            });
        }

        res.status(200).json({
            success: true,
            message: "Document deleted successfully",
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to delete document",
            error: error.message,
        });
    }
};

export const searchDocumentsByTags = async (req, res) => {
    try {
        const { tag } = req.params;

        const documents = await Document.find({
            tags: { $in: [tag] },
        });

        res.status(200).json({
            success: true,
            documents,
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to search documents by tags",
            error: error.message,
        });
    }
};