from src.common.utils import Settings
from src.datastore import CreateDataStore
from src.query_api import CDRCQuery


def main():
    query = CDRCQuery(**Settings().cdrc.model_dump())

    try:
        query.run()
    except Exception as e:
        print(e)
        print("Query failed to run. Check the logs for more information.")

    if query.files_changed:
        datastore = CreateDataStore(**Settings().datastore.model_dump())
        try:
            datastore.run()
        except Exception as e:
            print(e)
            print("Datastore failed to run. Check the logs for more information.")


if __name__ == "__main__":
    main()
