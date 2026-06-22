import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";

// ⚠️ Note: The correct legacy chain import paths are:
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";


export const handler = async (event) => {
  console.log("event : ", event.body);
  const userInput = JSON.parse(event.body).question;
  
  try {
    
    console.log("1. Initializing Pinecone and Embeddings...");
    const pinconeClient = new PineconeClient({
      apiKey: process.env.PINECONE_API_KEY,
    });
    
    const pineconeIndex = pinconeClient.Index("dsa-chatbot"); 

    // ध्यान दें: Model और Dimensions वही होने चाहिए जो अपलोड करते वक्त थे
    // Note: The Model and Dimensions must be the same as they were at the time of upload.
    const embeddings = new OpenAIEmbeddings({
      model: "text-embedding-3-small",
      dimensions: 1024, 
    });

    // Vector Store को लोड करें
    // Load the Vector Store
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
    });

    console.log("2. Setting up Retriever and LLM...");
    // Vector Store को Retriever में बदलें (यह top 4 सबसे रिलेवेंट डाक्यूमेंट्स लाएगा)
    // Convert the Vector Store into a Retriever (this will fetch the top 4 most relevant documents)
    const retriever = vectorStore.asRetriever({
      k: 4, 
    });

    // LLM (Chat Model) सेट करें
    // Set the LLM (Chat Model)
    const llm = new ChatOpenAI({
      model: "gpt-4o-mini",
      temperature: 0.3,    // कम टेम्परेचर से सटीक जवाब मिलते हैं Precise results are obtained at lower temperatures.
    });

    // 3. Prompt Template तैयार करें (AI को गाइड करने के लिए)
    // 3. Prepare a prompt template (to guide the AI)
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "You are a helpful assistant. Answer the user's question using only the provided context below. If you don't know the answer or if it's not in the context, say that you don't know.\n\nContext:\n{context}"],
      ["human", "{input}"],
    ]);

    console.log("3. Creating RAG Chain...");
    // Documents को कंबाइन करने वाली चेन
    // Chain that combines documents
    const combineDocsChain = await createStuffDocumentsChain({
      llm,
      prompt,
    });

    // फाइनल Retrieval Chain
    // Final Retrieval Chain
    const retrievalChain = await createRetrievalChain({
      retriever,
      combineDocsChain,
    });

    console.log(`4. Querying for: "${userInput}"`);
    // Chain को रन करें
    // Run the chain
    const response = await retrievalChain.invoke({
      input: userInput,
    });

    const answer = response.answer;
    // जवाब प्रिंट करें
    // Print the answer
    console.log("\n🤖 Answer:");
    console.log(response.answer);
    
    // वैकल्पिक: आप देख सकते हैं कि किस सोर्स डाक्यूमेंट्स से जवाब बना
    // console.log("\n📚 Sources used:", response.context.map(doc => doc.metadata));

     // TODO implement
    return {
      statusCode: 200,
      body: {answer},
    };

  } catch (error) {
    console.error("Error during query:", error);
  }
 
};
