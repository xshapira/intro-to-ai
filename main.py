import json

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate


def main() -> None:
    cloud_config = {"secure_connect_bundle": "./env/secure-connect-intro-to-ai.zip"}

    with open("./env/env.json") as fp:
        secrets = json.load(fp)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]
    ASTRA_DB_KEYSPACE = "db"
    OPENAI_API_KEY = secrets["openaiKey"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()

    message_history = CassandraChatMessageHistory(
        session_id="any",
        session=session,
        keyspace=ASTRA_DB_KEYSPACE,
        ttl_seconds=3600,
    )

    message_history.clear()

    cass_buff_memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
    )

    template = """
    You are now the captain of a ship sailing on the high seas in search of treasure. Your crew is looking for the legendary Treasure of the Lost Island, said to be worth a fortune beyond imagination. You must guide your crew through a series of challenges, choices, and consequences, dynamically adapting the story based on your crew's decisions. Your goal is to create a branching narrative experience where each choice leads to a new path, ultimately determining your crew's success.

    Here are some rules to follow:
    1. Start by asking the player to choose a type of ship that will be used later in the game.
    2. Have a few paths that lead to success.
    3. Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "The End.", I will search for this text to end the game


    Here is the chat history, use this to understand what to say next: {chat_history}
    Human: {human_input}
    AI:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template,
    )

    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=cass_buff_memory,
    )

    choice = "start"

    while True:
        response = llm_chain.predict(human_input=choice)
        print(response.strip())

        if "The End." in response:
            break

        choice = input("Your reply: ")
        # if choice == "quit":
        #     break


if __name__ == "__main__":
    main()
