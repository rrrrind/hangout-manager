class RemovePsFromUserPersonals < ActiveRecord::Migration[6.0]
  def change
    remove_column :user_personals, :p_four, :integer
    remove_column :user_personals, :p_five, :integer
    remove_column :user_personals, :p_six, :integer
  end
end
