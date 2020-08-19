class AddPsToUserPersonals < ActiveRecord::Migration[6.0]
  def change
    add_column :user_personals, :p_four, :integer
  end
end
