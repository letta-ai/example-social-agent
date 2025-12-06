from letta_client import Letta
from typing import Optional

def upsert_block(letta: Letta, label: str, value: str, **kwargs):
    """
    Ensures that a block by this label exists. If the block exists, it will
    replace content provided by kwargs with the values in this function call.
    """
    # Get the list of blocks (SDK v1.0 returns page object)
    blocks_page = letta.blocks.list(label=label)
    blocks = blocks_page.items if hasattr(blocks_page, 'items') else blocks_page

    # Check if we had any -- if not, create it
    if len(blocks) == 0:
        # Make the new block
        new_block = letta.blocks.create(
            label=label,
            value=value,
            **kwargs
        )

        return new_block

    if len(blocks) > 1:
        raise Exception(f"{len(blocks)} blocks by the label '{label}' retrieved, label must identify a unique block")
    
    else:
        existing_block = blocks[0]

        if kwargs.get('update', False):
            # Remove 'update' from kwargs before passing to update
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop('update', None)

            updated_block = letta.blocks.update(
                block_id = existing_block.id,
                label = label,
                value = value,
                **kwargs_copy
            )

            return updated_block
        else:
            return existing_block

def upsert_agent(letta: Letta, name: str, **kwargs):
    """
    Ensures that an agent by this label exists. If the agent exists, it will
    update the agent to match kwargs.
    """
    # Get the list of agents (SDK v1.0 returns page object)
    agents_page = letta.agents.list(name=name)
    agents = agents_page.items if hasattr(agents_page, 'items') else agents_page

    # Check if we had any -- if not, create it
    if len(agents) == 0:
        # Make the new agent
        new_agent = letta.agents.create(
            name=name,
            **kwargs
        )

        return new_agent

    if len(agents) > 1:
        raise Exception(f"{len(agents)} agents by the label '{label}' retrieved, label must identify a unique agent")
    
    else:
        existing_agent = agents[0]

        if kwargs.get('update', False):
            # Remove 'update' from kwargs before passing to update
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop('update', None)

            updated_agent = letta.agents.update(
                agent_id = existing_agent.id,
                **kwargs_copy
            )

            return updated_agent
        else:
            return existing_agent
        
        

    

    
    

    

    
    
