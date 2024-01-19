import discord


class LoginModal(discord.ui.Modal, title="Login"):
    api_key_input = discord.ui.TextInput(
        label="AI Horde API Key",
    )

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()


class LoginButtonView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.api_key: str | None = None

    @discord.ui.button(label="Login", style=discord.ButtonStyle.primary)
    async def login(self, interaction: discord.Interaction, _):
        modal = LoginModal()
        await interaction.response.send_modal(modal)
        await modal.wait()
        self.api_key = modal.api_key_input.value
        self.stop()
