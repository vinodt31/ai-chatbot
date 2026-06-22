import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";

async function indexDocumentToVectorStore() {
  console.log("1. Loading PDF...");
  const loader = new PDFLoader("./about-DSA.pdf");
  const docs = await loader.load();

  console.log("2. Splitting text into chunks...");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splitDocs = await splitter.splitDocuments(docs);

  console.log("3. Initializing Pinecone Client...");
  const pinconeClient = new PineconeClient({
    apiKey: process.env.PINECONE_API_KEY,
  });
  
  // Enter the name of the index you created on the Pinecone dashboard here.
  const pineconeIndex = pinconeClient.Index("dsa-chatbot"); 

  console.log("4. Converting to embeddings and uploading to Pinecone Cloud...");
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    dimensions: 1024, 
  });

  /** 
  The PineconeStore.fromDocuments() function does both. It is responsible for both converting the chunks into embeddings 
  AND saving them into the Pinecone database.It handles the entire pipeline for you in one single step.
  */
  await PineconeStore.fromDocuments(splitDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5, // For better upload speed
  });

  console.log("✅ Success: Data successfully saved to Pinecone Cloud!");
}

indexDocumentToVectorStore().catch(console.error);