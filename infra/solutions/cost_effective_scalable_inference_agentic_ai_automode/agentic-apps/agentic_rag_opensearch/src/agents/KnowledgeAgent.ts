import fs from "fs";
import path from "path";
import crypto from "crypto";
import EmbeddingRetriever from "../EmbeddingRetriever";
import { logTitle } from "../utils";

interface FileMetadata {
    path: string;
    hash: string;
    lastModified: number;
    size: number;
}

export default class KnowledgeAgent {
    private knowledgePath: string;
    private metadataPath: string;
    private embeddingRetriever: EmbeddingRetriever | null = null;
    private lastKnownState: Map<string, FileMetadata> = new Map();

    constructor() {
        this.knowledgePath = path.join(process.cwd(), 'knowledge');
        this.metadataPath = path.join(process.cwd(), '.knowledge-metadata.json');
    }

    async init() {
        logTitle('INITIALIZING KNOWLEDGE AGENT');
        
        // Initialize embedding retriever with llamacpp-embedding model
        this.embeddingRetriever = new EmbeddingRetriever("llamacpp-embedding");
        
        // Load existing metadata if available
        await this.loadMetadata();
        
        console.log('Knowledge Agent initialized');
    }

    async close() {
        // Save metadata before closing
        await this.saveMetadata();
        
        // Close embedding retriever if needed
        if (this.embeddingRetriever) {
            // @ts-ignore - Access private property for cleanup
            if (this.embeddingRetriever.vectorStore && typeof this.embeddingRetriever.vectorStore.close === 'function') {
                // @ts-ignore
                await this.embeddingRetriever.vectorStore.close();
            }
        }
    }

    async checkForChanges(): Promise<boolean> {
        logTitle('CHECKING KNOWLEDGE CHANGES');
        
        if (!fs.existsSync(this.knowledgePath)) {
            console.log(`Knowledge directory not found: ${this.knowledgePath}`);
            return false;
        }

        const currentState = await this.scanKnowledgeDirectory();
        const hasChanges = this.compareStates(currentState);
        
        if (hasChanges) {
            console.log('Changes detected in knowledge files');
            this.lastKnownState = currentState;
        } else {
            console.log('No changes detected in knowledge files');
        }
        
        return hasChanges;
    }

    async embedKnowledge(): Promise<boolean> {
        logTitle('EMBEDDING KNOWLEDGE FILES');
        
        if (!this.embeddingRetriever) {
            console.error('Embedding retriever not initialized');
            return false;
        }

        try {
            // Get all knowledge files (including CSV)
            const files = this.getAllKnowledgeFiles();
            console.log(`Found ${files.length} knowledge files to embed`);
            
            let successCount = 0;
            
            // Process each file
            for (const file of files) {
                try {
                    console.log(`Processing: ${path.basename(file)}`);
                    
                    const ext = path.extname(file).toLowerCase();
                    
                    if (ext === '.csv') {
                        // Handle CSV files specially
                        const csvSuccess = await this.processCsvFile(file);
                        if (csvSuccess) successCount++;
                    } else {
                        // Handle regular text files
                        const content = fs.readFileSync(file, 'utf-8');
                        const enrichedContent = `File: ${path.basename(file)}\n\n${content}`;
                        await this.embeddingRetriever.embedDocument(enrichedContent);
                        successCount++;
                    }
                } catch (error) {
                    console.error(`Error processing file ${file}:`, error);
                }
            }
            
            console.log(`Successfully embedded ${successCount}/${files.length} files`);
            
            // Save current state as metadata
            await this.saveMetadata();
            
            return successCount > 0;
        } catch (error) {
            console.error('Error in knowledge embedding:', error);
            return false;
        }
    }

    private async scanKnowledgeDirectory(): Promise<Map<string, FileMetadata>> {
        const state = new Map<string, FileMetadata>();
        
        if (!fs.existsSync(this.knowledgePath)) {
            return state;
        }
        
        const files = this.getAllKnowledgeFiles();
        
        for (const file of files) {
            try {
                const stats = fs.statSync(file);
                const content = fs.readFileSync(file, 'utf-8');
                const hash = crypto.createHash('md5').update(content).digest('hex');
                
                state.set(file, {
                    path: file,
                    hash,
                    lastModified: stats.mtime.getTime(),
                    size: stats.size
                });
            } catch (error) {
                console.error(`Error scanning file ${file}:`, error);
            }
        }
        
        return state;
    }

    private compareStates(currentState: Map<string, FileMetadata>): boolean {
        // If no previous state, consider it as changes
        if (this.lastKnownState.size === 0) {
            return currentState.size > 0;
        }
        
        // Check for new or modified files
        for (const [filePath, metadata] of currentState) {
            const previousMetadata = this.lastKnownState.get(filePath);
            
            if (!previousMetadata || previousMetadata.hash !== metadata.hash) {
                return true;
            }
        }
        
        // Check for deleted files
        for (const filePath of this.lastKnownState.keys()) {
            if (!currentState.has(filePath)) {
                return true;
            }
        }
        
        return false;
    }

    private getAllKnowledgeFiles(): string[] {
        if (!fs.existsSync(this.knowledgePath)) {
            return [];
        }
        
        const files: string[] = [];
        const entries = fs.readdirSync(this.knowledgePath, { withFileTypes: true });
        
        for (const entry of entries) {
            if (entry.isFile()) {
                const filePath = path.join(this.knowledgePath, entry.name);
                const ext = path.extname(entry.name).toLowerCase();
                
                // Support all knowledge file types including CSV
                if (['.md', '.txt', '.json', '.csv'].includes(ext)) {
                    files.push(filePath);
                }
            }
        }
        
        return files;
    }

    private async processCsvFile(filePath: string): Promise<boolean> {
        try {
            const content = fs.readFileSync(filePath, 'utf-8');
            const rows = content.split('\n').filter(row => row.trim());
            
            if (rows.length > 1) {
                const headers = rows[0].split(',');
                let processedRows = 0;
                
                // Process each row as a separate document
                for (let i = 1; i < rows.length; i++) {
                    const values = rows[i].split(',');
                    const rowData = headers.map((header, index) => 
                        `${header.trim()}: ${values[index]?.trim() || ''}`
                    ).join('\n');
                    
                    const enrichedContent = `CSV File: ${path.basename(filePath)}\nRow ${i}:\n${rowData}`;
                    await this.embeddingRetriever!.embedDocument(enrichedContent);
                    processedRows++;
                }
                
                console.log(`Processed ${processedRows} rows from CSV file`);
                return true;
            }
            
            return false;
        } catch (error) {
            console.error(`Error processing CSV file ${filePath}:`, error);
            return false;
        }
    }

    private getKnowledgeFiles(): string[] {
        if (!fs.existsSync(this.knowledgePath)) {
            return [];
        }
        
        const files: string[] = [];
        const entries = fs.readdirSync(this.knowledgePath, { withFileTypes: true });
        
        for (const entry of entries) {
            if (entry.isFile()) {
                const filePath = path.join(this.knowledgePath, entry.name);
                const ext = path.extname(entry.name).toLowerCase();
                
                if (['.md', '.txt', '.json'].includes(ext)) {
                    files.push(filePath);
                }
            }
        }
        
        return files;
    }

    private async loadMetadata() {
        try {
            if (fs.existsSync(this.metadataPath)) {
                const data = fs.readFileSync(this.metadataPath, 'utf-8');
                const metadata = JSON.parse(data);
                
                this.lastKnownState = new Map(Object.entries(metadata));
                console.log(`Loaded metadata for ${this.lastKnownState.size} files`);
            }
        } catch (error) {
            console.error('Error loading metadata:', error);
        }
    }

    private async saveMetadata() {
        try {
            const metadata = Object.fromEntries(this.lastKnownState);
            fs.writeFileSync(this.metadataPath, JSON.stringify(metadata, null, 2));
            console.log(`Saved metadata for ${this.lastKnownState.size} files`);
        } catch (error) {
            console.error('Error saving metadata:', error);
        }
    }
}
